import numpy as np
import pandas as pd
from scipy.linalg import eig
from scipy.optimize import linear_sum_assignment
from scipy.stats import norm

##########################
# Helper Functions
##########################
def _convert_spec(mzs, pmzs=None, nl_trim=1.5):
    spec_ids = np.concatenate([[n] * len(m) for n, m in enumerate(mzs)])
    if pmzs is None:
        pmzs = np.array([np.nan] * len(mzs))
    pmzs = np.asarray(pmzs)[np.concatenate([[n] * len(m) for n, m in enumerate(mzs)])]
    mzs = np.concatenate(mzs)
    nls = pmzs - mzs
    nls[nls < nl_trim] = np.nan

    mz_diffs = np.subtract.outer(mzs, mzs)
    nl_diffs = np.subtract.outer(nls, nls)

    return mzs, nls, pmzs, spec_ids, mz_diffs, nl_diffs


##########################
# Spectral Graph Functions
##########################
def _counts_matrix(D, tolerance):
    d = D.ravel()
    sort_idx = np.argsort(d)

    left = np.searchsorted(d[sort_idx], d - tolerance / 2, "left")
    right = np.searchsorted(d[sort_idx], d + tolerance / 2, "right")

    c = right - left
    c[np.isnan(d)] = 0

    C = c.reshape(D.shape)

    return C


def _transition_matrix(W):
    DI = np.diag(1.0 / W.sum(axis=1))
    P = DI.dot(W)

    return P


def _stationary_distribution(P):
    evals, evecs = eig(P.T)
    evecs = evecs[:, 0]

    v = evecs.flatten().real
    p = v / v.sum()

    return p


def _diplacian(W):
    P = _transition_matrix(W)

    p = _stationary_distribution(P)
    p = np.repeat(p, len(p)).reshape(-1, len(p))

    D = P - p

    return D, p


def _sym_norm_laplacian(W):
    D, p = _diplacian(W)
    L = (D + D.T) / 2

    return L, p


##########################
# Similarity Functions
##########################


def similarity_matrix(mzs, pmzs=None, tolerance=0.01, nl_trim=1.5, iters=2):
    """
    Return fragmentation similarity matrix, S,
    and spectrum id for each row/column, spec_ids
    using list of mz numpy arrays, mzs
    list of precursor mz values, pmzs,
    max number of Da between equivalent mzs and nls, tolerance,
    and min Da for nls, nl_trim (to reduce bias from pmz nls always being 0)
    """

    mzs, nls, pmzs, spec_ids, mz_diffs, nl_diffs = _convert_spec(mzs, pmzs, nl_trim)

    C = np.zeros_like(mz_diffs, dtype=float)
    for n in set(spec_ids):
        mz_count = _counts_matrix(mz_diffs[spec_ids == n], tolerance)
        nl_count = _counts_matrix(nl_diffs[spec_ids == n], tolerance)
        C[spec_ids == n] = mz_count + nl_count + (mz_count * nl_count) ** 0.5

    L, p = _sym_norm_laplacian(C)

    S = -p
    for i in range(1, iters + 1):
        S += L
        if i < iters:
            L = L.dot(L)

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

    match_scores = S[(tuple(row_ind), tuple(col_ind))]
    good_match = match_scores > 0

    M[(tuple(row_ind[good_match]), tuple(col_ind[good_match]))] = 1

    return M


def multiple_match(S, spec_ids):
    """
    Return max weight matching matrix, M, of simile score matrix, S
    with rows treated seperately accoring to their spectrum id, spec_ids
    """

    M = np.zeros_like(S)
    for n in set(spec_ids):
        M[spec_ids == n] = pairwise_match(S[spec_ids == n])

    return M


def inter_intra_compare(spec_ids):
    """
    Returns pro/con comparison matrix, C,
    such that interspectral comparisons are 1 (pro)
    and intraspectral comparisions are -1 (con)
    using spectrum ids, spec_ids, to deliniate spectra
    """

    C = 2 * np.not_equal.outer(spec_ids, spec_ids) - 1

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

    scores = (S * C * M).sum(0)
    probs = (C > 0).mean(0)

    _, length = np.unique(spec_ids, return_counts=True)

    index = np.concatenate([np.arange(l) for l in length])
    start = np.arange(len(index)) - index
    length = np.concatenate([[l] * l for l in length])

    shifts = (
        np.add.outer(np.arange(-gap_penalty, gap_penalty + 1), index + length) % length
    ) + start

    scores = scores[shifts].mean(0)
    probs = probs[shifts].mean(0)

    score = scores.clip(0).sum() / abs(scores).sum()

    return score, scores, probs


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

    comparisons = 2 * (rng.random((iterations, len(scores))) <= probs) - 1
    null_dist = comparisons.dot(abs(scores))

    return null_dist


def mcp_test(scores, probs, log_size=5, return_dist=False, early_stop=False, seed=None):
    """
    Return approximation of 2D Monte Carlo permutation test
    using match score of each fragment ion, scores,
    following pro/con comparison probablilities, probs
    """

    assert isinstance(log_size, int)
    log_size = max(log_size, 2)

    score = scores.sum()

    null_dist = []
    pval = 1.0
    start = 2 if early_stop else log_size
    for log_iter in range(start, log_size + 1):
        iterations = 10**log_iter

        null_dist.extend(
            null_distribution(scores, probs, iterations - len(null_dist), seed)
        )

        # Subtract off miniscule amount for floating point error
        new_pval = (score - 1e-9 <= np.array(null_dist)).sum() / iterations
        new_pval = max(new_pval, 1.0 / iterations)

        if (new_pval / pval) <= 0.9**log_iter:
            pval = new_pval
        else:
            pval = new_pval
            break

    return (pval, np.array(null_dist)) if return_dist else pval


def z_test(scores, probs, log_size=6, return_dist=False, seed=None):
    """
    Return approximation of z-test using
    match score of each fragment ion, scores,
    following pro/con comparison probablilities, probs
    """

    assert isinstance(log_size, int)
    log_pop_size = max(log_size, 5)

    score = scores.sum()

    null_dist = null_distribution(scores, probs, 10**log_size, seed)

    z_score = (score - null_dist.mean()) / null_dist.std()
    pval = norm.sf(z_score)

    return (pval, np.array(null_dist)) if return_dist else pval


##########################
# Analysis Functions
##########################
def matching_ions_report(
    S, C, M, mzs, pmzs=None, spec_name=None, comp_types=["con", "none", "pro"]
):
    """
    Return matching ions report DataFrame, mi_df,
    describing all matching ions found using
    S, C, M, mzs, and pmzs
    with option to name mass spectra used, spec_name
    """
    match_idxs = np.where(np.triu(np.maximum(M, M.T)))

    mzs, nls, pmzs, spec_ids, mz_diffs, nl_diffs = _convert_spec(mzs, pmzs)

    if spec_name is None:
        spec_key = spec_ids
    else:
        spec_key = np.asarray(spec_name)[spec_ids]

    mi_df = {}
    for name, var in [("spec_key", spec_key), ("pmz", pmzs), ("mz", mzs)]:
        mi_df[name + "_1"] = var[match_idxs[0]]
        mi_df[name + "_2"] = var[match_idxs[1]]
    for name, var in [
        ("mz_diff", mz_diffs),
        ("nl_diff", nl_diffs),
        ("score", S),
        ("type", np.asarray(comp_types)[C.astype(int) + 1]),
    ]:
        mi_df[name] = var[match_idxs]

    mi_df = pd.DataFrame(mi_df)

    return mi_df
