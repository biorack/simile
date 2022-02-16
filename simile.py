import bisect
import numpy as np
import pandas as pd
from scipy.linalg import eig
from scipy.optimize import linear_sum_assignment
from scipy.stats import norm

##########################
# Helper Functions
##########################
def _convert_spec(mzs, pmzs=None, nl_trim=1.5):
    spec_ids = np.concatenate([[n]*len(m) for n,m in enumerate(mzs)])
    if pmzs is None:
        pmzs = np.array([np.nan]*len(mzs))
    pmzs = np.asarray(pmzs)[np.concatenate([[n]*len(m) for n,m in enumerate(mzs)])]
    mzs = np.concatenate(mzs)
    nls = pmzs - mzs
    nls[nls < nl_trim] = np.nan

    mz_diffs = np.subtract.outer(mzs,mzs)
    nl_diffs = np.subtract.outer(nls,nls)

    return mzs, nls, pmzs, spec_ids, mz_diffs, nl_diffs


##########################
# Spectral Graph Functions
##########################
def _transition_matrix(W):
    DI = np.diag(1.0/W.sum(axis=1))
    P = DI.dot(W)

    return P

def _stationary_distribution(P):
    evals, evecs = eig(P.T)
    evecs = evecs[:,0]

    v = evecs.flatten().real
    p =  v/v.sum()

    return p

def _diplacian(W):
    I = np.identity(W.shape[0])

    P = _transition_matrix(W)

    p = _stationary_distribution(P)

    sqrt_p = np.sqrt(p)

    D = np.diag(sqrt_p).dot(I-P).dot(np.diag(1.0/sqrt_p))

    return D

def _sym_norm_laplacian(W):
    D = _diplacian(W)
    L = (D+D.T)/2

    return L


##########################
# Similarity Functions
##########################
def _counts_matrix(frag_diffs, tolerance, tol_weight=.5):
    """
    Return fragment difference counts matrix, frag_counts,
    using fragment difference matrix, frag_diffs,
    max number of Da between equivalent frag differences, tolerance,
    and power weighting factor inversely proportional to tolerance, tol_weight
    (which only matters when using multiple tolerances)
    """

    if not (isinstance(tolerance, list) or isinstance(tolerance, np.ndarray)):
        tolerance = [tolerance]
    tolerance = sorted(tolerance, reverse=True)


    frag_counts = np.zeros_like(frag_diffs, dtype=float)
    frag_diffs = np.array([(*idx, val) for idx, val
                         in np.ndenumerate(frag_diffs)
                         if not np.isnan(val)])
    if frag_diffs.size:
        frag_diffs = frag_diffs[np.argsort(frag_diffs[:,2])]

        for i in range(len(frag_diffs)):
            left_idx = 0
            right_idx = -1
            for tol in tolerance:
                left_idx = bisect.bisect_right(frag_diffs[:,2], frag_diffs[i,2]-(tol/2),
                                               lo=left_idx, hi=right_idx)
                right_idx = bisect.bisect_left(frag_diffs[:,2], frag_diffs[i,2]+(tol/2),
                                               lo=left_idx, hi=right_idx)

                count = right_idx - left_idx

                frag_counts[int(frag_diffs[i,0]),int(frag_diffs[i,1])] += count/(tol**tol_weight)

    return frag_counts

def similarity_matrix(mzs, pmzs=None, tolerance=.01, nl_trim=1.5):
    """
    Return fragmentation similarity matrix, S,
    and spectrum id for each row/column, spec_ids
    using list of mz numpy arrays, mzs
    list of precursor mz values, pmzs,
    max number of Da between equivalent mzs and nls, tolerance,
    and min Da for nls, nl_trim (to reduce bias from pmz nls always being 0)
    """

    mzs, nls, pmzs, spec_ids, mz_diffs, nl_diffs = _convert_spec(mzs, pmzs, nl_trim)

    frag_count = np.zeros_like(mz_diffs, dtype=float)
    for n in set(spec_ids):
        mz_count = _counts_matrix(mz_diffs[spec_ids==n], tolerance)
        nl_count = _counts_matrix(nl_diffs[spec_ids==n], tolerance)
        frag_count[spec_ids==n] = mz_count+nl_count+(mz_count*nl_count)**.5

    S = np.linalg.pinv(_sym_norm_laplacian(frag_count))
    S -= np.diag(S.diagonal())

    return S, spec_ids

##########################
# Comparison Functions
##########################
def pairwise_match(S):
    """
    Return max weight matching matrix, M, of simile score matrix, S
    """

    M = np.zeros_like(S)
    row_ind, col_ind = linear_sum_assignment(S, maximize=True)

    match_scores = S[(tuple(row_ind),tuple(col_ind))]
    good_match = match_scores > 0

    M[(tuple(row_ind[good_match]),tuple(col_ind[good_match]))] = 1

    return M

def multiple_match(S, spec_ids):
    """
    Return max weight matching matrix, M, of simile score matrix, S
    with rows treated seperately accoring to their spectrum id, spec_ids
    """

    M = np.zeros_like(S)
    for n in set(spec_ids):
        M[spec_ids==n] = pairwise_match(S[spec_ids==n])

    return M

def inter_intra_compare(spec_ids):
    """
    Returns pro/con comparison matrix, C,
    such that interspectral comparisons are 1 (pro)
    and intraspectral comparisions are -1 (con)
    using spectrum ids, spec_ids, to deliniate spectra
    """

    C = 2*np.not_equal.outer(spec_ids,spec_ids)-1

    return C

def match_scores(S, C, M, spec_ids, gap_penalty):
    """
    Return match score, scores, and pro/con comparison probablility, probs,
    of each fragment ion as flattened sum of products of
    simile score matrix, S,
    max weight matching matrix, M,
    and comparison matrix, C
    with intra-spectral rolling average of +/- window size, gap_penalty,
    using spectrum ids, spec_ids, to deliniate spectra
    """

    scores = (S*C*M).sum(0)
    probs = (C>0).mean(0)

    _,length = np.unique(spec_ids, return_counts=True)

    index = np.concatenate([np.arange(l) for l in length])
    start = np.arange(len(index))-index
    length = np.concatenate([[l]*l for l in length])

    shifts = (np.add.outer(np.arange(-gap_penalty, gap_penalty+1),index+length)%length)+start

    scores = scores[shifts].mean(0)
    probs = probs[shifts].mean(0)

    return scores, probs


##########################
# Statistics Functions
##########################
def null_distribution(scores, probs, iterations=1e5, seed=None):
    """
    Return null distribution, null_dist, of size iterations
    using match score of each fragment ion, scores,
    following pro/con comparison probablilities, probs
    """

    rng = np.random.default_rng(seed)


    comparisons = 2*(rng.random((iterations,len(scores))) <= probs)-1
    null_dist = comparisons.dot(abs(scores))

    return null_dist

def mcp_test(scores, probs, log_size=5, return_dist=False, early_stop=False, seed=None):
    """
    Return approximation of 2D Monte Carlo permutation test using
    using match score of each fragment ion, scores,
    following pro/con comparison probablilities, probs
    """

    assert isinstance(log_size, int)
    log_size = max(log_size, 2)

    score = scores.sum()

    null_dist = []
    pval = 1.0
    start = 2 if early_stop else log_size
    for log_iter in range(start, log_size+1):
        iterations = 10**log_iter

        null_dist.extend(null_distribution(scores, probs, iterations-len(null_dist), seed))

        # Subtract off miniscule amount for floating point error
        new_pval = (score-1e-9 <= np.array(null_dist)).sum()/iterations
        new_pval = max(new_pval, 1.0/iterations)

        if (new_pval/pval) <= .9**log_iter:
            pval = new_pval
        else:
            pval = new_pval
            break

    return (pval, np.array(null_dist)) if return_dist else pval

def z_test(scores, probs, log_size=6, return_dist=False, seed=None):
    """
    Return approximation of z-test using using
    using match score of each fragment ion, scores,
    following pro/con comparison probablilities, probs
    """

    assert isinstance(log_size, int)
    log_pop_size = max(log_size, 5)

    score = scores.sum()

    null_dist = null_distribution(scores, probs, 10**log_size, seed)

    z_score = (score-null_dist.mean())/null_dist.std()
    pval = norm.sf(z_score)

    return (pval, np.array(null_dist)) if return_dist else pval


##########################
# Analysis Functions
##########################
def matching_ions_report(S, C, M, mzs, pmzs=None, spec_name=None):
    """
    Return matching ions report DataFrame, mi_df,
    describing all matching ions found using
    S, C, M, mzs, and pmzs
    with option to name mass spectra used, spec_name
    """
    match_idxs = np.where(np.triu(np.maximum(M,M.T)))

    mzs, nls, pmzs, spec_ids, mz_diffs, nl_diffs = _convert_spec(mzs, pmzs)

    if spec_name is None:
        spec_key = spec_ids
    else:
        spec_key = np.asarray(spec_name)[spec_ids]

    mi_df = {}
    for name,var in [('spec_key', spec_key), ('pmz', pmzs), ('mz', mzs)]:
        mi_df[name+'_1'] = var[match_idxs[0]]
        mi_df[name+'_2'] = var[match_idxs[1]]
    for name,var in [('mz_diff', mz_diffs), ('nl_diff', nl_diffs), ('score', S),
                     ('type', np.array(['none','pro','con'])[C.astype(int)])]:
        mi_df[name] = var[match_idxs]

    mi_df = pd.DataFrame(mi_df)

    return mi_df
