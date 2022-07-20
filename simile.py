import numpy as np
import pandas as pd
from scipy.linalg import eig
from scipy.optimize import linear_sum_assignment
from scipy.stats import norm
from scipy import sparse as sp

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
    for n in np.unique(spec_ids):
        mz_count = _counts_matrix(mz_diffs[spec_ids == n], tolerance)
        nl_count = _counts_matrix(nl_diffs[spec_ids == n], tolerance)
        C[spec_ids == n] = mz_count + nl_count + (mz_count * nl_count) ** 0.5

    L, p = _sym_norm_laplacian(C)

    S = np.sum([np.linalg.matrix_power(L, i) for i in range(1, iters + 1)], axis=0) - p
    S -= np.diag(S.diagonal())

    return S, spec_ids


##########################
# Comparison Functions
##########################
def pairwise_match(S):
    """
    Return max weight matching matrix, M, of simile score matrix, S
    """

    row, col = linear_sum_assignment(S, maximize=True)

    M = sp.coo_matrix(
        (S[tuple(row), tuple(col)], (row, col)), shape=S.shape, dtype=bool
    )

    return M


def multiple_match(S, spec_ids):
    """
    Return max weight matching matrix, M, of simile score matrix, S
    with rows treated seperately accoring to their spectrum id, spec_ids
    """

    M = sp.vstack([pairwise_match(S[spec_ids == n]) for n in np.unique(spec_ids)])

    return M


def sym_compare(M, spec_ids):
    """
    Returns pro/con comparison matrix, C,
    such that symmetric matches are 1 (pro)
    and asymmetric matches are -1 (con)
    using spectrum ids, spec_ids, to deliniate spectra
    """

    C = M.toarray().dot(np.equal.outer(spec_ids, spec_ids))
    C &= np.not_equal.outer(spec_ids, spec_ids)
    C = 2 * (C | C.T) - 1

    return C


def inter_intra_compare(spec_ids):
    """
    Returns pro/con comparison matrix, C,
    such that interspectral matches are 1 (pro)
    and intraspectral matches are -1 (con)
    using spectrum ids, spec_ids, to deliniate spectra
    """

    C = 2 * np.not_equal.outer(spec_ids, spec_ids) - 1

    return C


def match_scores(S, C, M):
    """
    Return fragment match scores, frag_scores,
    and pro/con comparison probablility, frag_probs,
    of each fragment ion as elementwise products of
    simile score matrix, S,
    max weight matching matrix, M,
    and comparison matrix, C
    """

    frag_scores = M.multiply(S * C)
    frag_probs = (C > 0).mean(axis=0)

    return frag_scores, frag_probs


##########################
# Statistics Functions
##########################
def null_distribution(frag_scores, frag_probs, iterations=1e5, seed=None):
    """
    Return null distribution, null_dist, of size iterations
    using match score of each fragment ion, frag_scores,
    following pro/con comparison probablilities, frag_probs
    """

    rng = np.random.default_rng(seed)

    comparisons = 2 * (rng.random((iterations, len(frag_scores))) <= frag_probs) - 1
    null_dist = comparisons * frag_scores

    return null_dist


def mcp_test(
    S,
    M,
    C,
    spec_ids,
    log_size=5,
    return_dist=False,
    seed=None,
):
    """
    Return approximation of 2D Monte Carlo permutation test
    using simile score matrix, S,
    max weight matching matrix, M,
    and comparison matrix, C
    delineating between spectra with spec_ids
    """

    assert isinstance(log_size, int)
    log_size = max(log_size, 2)

    frag_scores, frag_probs = match_scores(S, C, M)

    frag_to_spec = sp.coo_matrix(
        (np.ones_like(spec_ids), (np.arange(len(spec_ids)), spec_ids)),
    )

    spec_scores = frag_to_spec.T.dot(frag_scores).dot(frag_to_spec).toarray()
    spec_scores = (sp.triu(spec_scores) + sp.triu(spec_scores.T, 1)).tocoo()

    idx_sort = np.argsort(spec_scores.row + 1j * spec_scores.col)
    new_col = np.searchsorted(
        (spec_scores.row + 1j * spec_scores.col)[idx_sort],
        spec_ids[np.minimum(frag_scores.row, frag_scores.col)]
        + 1j * spec_ids[np.maximum(frag_scores.row, frag_scores.col)],
    )
    new_col = np.argsort(idx_sort)[new_col]

    frag_to_spec = sp.coo_matrix(
        (frag_to_spec.data, (np.arange(len(spec_ids)), new_col))
    )

    null_dist = null_distribution(frag_scores.data, frag_probs, 10**log_size, seed)
    null_dist = null_dist.dot(frag_to_spec.toarray())

    # Subtract off miniscule amount for floating point error
    pval = spec_scores.copy()
    pval.data = (spec_scores.data - 1e-9 <= np.array(null_dist)).sum(axis=0).clip(1) / (
        10**log_size
    )

    return (
        (spec_scores, pval, np.array(null_dist)) if return_dist else (spec_scores, pval)
    )


def z_test(
    S,
    M,
    C,
    spec_ids,
    log_size=6,
    return_dist=False,
    seed=None,
):
    """
    Return approximation of z-test using
    simile score matrix, S,
    max weight matching matrix, M,
    and comparison matrix, C
    delineating between spectra with spec_ids
    """

    assert isinstance(log_size, int)
    log_pop_size = max(log_size, 5)

    frag_scores, frag_probs = match_scores(S, C, M)

    frag_to_spec = sp.coo_matrix(
        (np.ones_like(spec_ids), (np.arange(len(spec_ids)), spec_ids)),
    )

    spec_scores = frag_to_spec.T.dot(frag_scores).dot(frag_to_spec).toarray()
    spec_scores = (sp.triu(spec_scores) + sp.triu(spec_scores.T, 1)).tocoo()

    idx_sort = np.argsort(spec_scores.row + 1j * spec_scores.col)
    new_col = np.searchsorted(
        (spec_scores.row + 1j * spec_scores.col)[idx_sort],
        spec_ids[np.minimum(frag_scores.row, frag_scores.col)]
        + 1j * spec_ids[np.maximum(frag_scores.row, frag_scores.col)],
    )
    new_col = np.argsort(idx_sort)[new_col]

    frag_to_spec = sp.coo_matrix(
        (frag_to_spec.data, (np.arange(len(spec_ids)), new_col))
    )

    null_dist = null_distribution(frag_scores.data, frag_probs, 10**log_size, seed)
    null_dist = null_dist.dot(frag_to_spec.toarray())

    z_score = (spec_scores.data - null_dist.mean(axis=0)) / null_dist.std(axis=0)
    pval = spec_scores.copy()
    pval.data = norm.sf(z_score)

    return (
        (spec_scores, pval, np.array(null_dist)) if return_dist else (spec_scores, pval)
    )


##########################
# Analysis Functions
##########################
def matching_ions_report(
    S, M, C, mzs, pmzs=None, spec_name=None, comp_types=["con", "none", "pro"]
):
    """
    Return matching ions report DataFrame, mi_df,
    describing all matching ions found using
    S, C, M, mzs, and pmzs
    with option to name mass spectra used, spec_name
    """
    M = sp.triu(M + M.T)

    mzs, nls, pmzs, spec_ids, mz_diffs, nl_diffs = _convert_spec(mzs, pmzs)

    if spec_name is None:
        spec_key = spec_ids
    else:
        spec_key = np.asarray(spec_name)[spec_ids]

    mi_df = {}
    for name, var in [("spec_key", spec_key), ("pmz", pmzs), ("mz", mzs)]:
        mi_df[name + "_1"] = var[M.row]
        mi_df[name + "_2"] = var[M.col]
    for name, var in [
        ("mz_diff", mz_diffs),
        ("nl_diff", nl_diffs),
        ("score", S),
        ("type", np.asarray(comp_types)[C.astype(int) + 1]),
    ]:
        mi_df[name] = var[M.row, M.col]

    mi_df = pd.DataFrame(mi_df)

    return mi_df
