import numpy as np
from scipy.linalg import eig
from scipy.optimize import linear_sum_assignment
from sortedcollections import SortedList, SortedSet

#########################
#Spectral Graph Functions
#########################
def transition_matrix(W):
    DI = np.diag(1.0/W.sum(axis=1))
    P = DI.dot(W)

    return P

def stationary_distribution(P):
    evals, evecs = eig(P.T)
    evecs = evecs[:, np.argmax(evals)]

    v = evecs.flatten().real
    p =  v / v.sum()

    return p

def laplacian(W):
    n, m = W.shape
    I = np.identity(n)

    P = transition_matrix(W)
    p = stationary_distribution(P)
    sqrt_p = np.sqrt(p)

    L = np.diag(sqrt_p).dot(I - P).dot(np.diag(1.0/sqrt_p))

    return L

def sym_norm_laplacian(W):
    L = laplacian(W)

    return (L+L.T)/2


###############################
# Similarity Matrix Functions
###############################
def counts_matrix(A, tolerance=.01):


    if isinstance(A, list):
        sorted_elements = SortedList(key=lambda i:i[1])
        for a in A:
            sorted_elements.update(np.ndenumerate(a))
        counts = np.zeros_like(a, dtype=float)

    else:
        sorted_elements = SortedList(np.ndenumerate(A), key=lambda i:i[1])
        counts = np.zeros_like(A, dtype=float)

    for (i,j),v in sorted_elements:
        left =  ((None,None), v - (tolerance/2))
        right = ((None,None), v + (tolerance/2))

        counts[i,j] += sorted_elements.bisect_right(right) - \
                       sorted_elements.bisect_left(left)

    return counts

def similarity_matrix(mzs1, mzs2, pmz1=None, pmz2=None, tolerance=.01):
    mzs = np.concatenate([mzs1,mzs2])
    mz_diffs = np.subtract.outer(mzs,mzs)

    frag_count = np.zeros_like(mz_diffs)

    frag_count[:len(mzs1)] = counts_matrix(mz_diffs[:len(mzs1)], tolerance)
    frag_count[len(mzs1):] = counts_matrix(mz_diffs[len(mzs1):], tolerance)

    if pmz1 is not None and pmz2 is not None:
        nls = np.concatenate([pmz1-mzs1,pmz2-mzs2])
        nl_diffs = np.subtract.outer(nls,nls)

        frag_count[:len(mzs1)] *= counts_matrix(nl_diffs[:len(mzs1)], tolerance)
        frag_count[len(mzs1):] *= counts_matrix(nl_diffs[len(mzs1):], tolerance)
        frag_count = frag_count**.5


    frag_sim = np.linalg.pinv(sym_norm_laplacian(frag_count), hermitian=False)

    return frag_sim


#####################
# Alignment Functions
#####################
def pairwise_align(S):
    A = np.zeros((S.shape[0]+1,S.shape[1]+1),dtype=float)
    B = np.zeros((3,S.shape[0]+1,S.shape[1]+1),dtype=int)
    B[:-1] = -1

    for i in range(1, A.shape[0]):
        for j in range(1, A.shape[1]):
            A[i,j],B[:,i,j] = max([A[i-1,j-1] + S[i-1,j-1], [i-1, j-1,  0]],
                                  [A[i-1,  j],              [i-1, j,    1]],
                                  [A[i,  j-1],              [i,   j-1,  1]],
                                  key=lambda a: a[0])

    idx = np.array([A.shape[0]-1, A.shape[1]-1])
    score = A[idx[0],idx[1]]

    alignment = []
    while (idx >= 0).any():
        if B[2,idx[0],idx[1]] == 0:
            alignment.append(idx[::-1])
        idx = B[:-1,idx[0],idx[1]]
    alignment = np.array(alignment)[:-1]-1

    return score, alignment

def pairwise_match(S):
    S = S.clip(0)

    col_match, row_match = linear_sum_assignment(S, maximize=True)

    match_scores = S[[tuple(col_match),tuple(row_match)]]
    score = match_scores.sum()
    good_match = match_scores > 0
    matches = np.asarray([row_match[good_match],col_match[good_match]]).T

    return score, matches

def fast_pairwise_alignment_score(S):
    if S.shape[0] < S.shape[1]:
        s = np.zeros(tuple([S.shape[0]+1])+S.shape[2:])

        for i in range(0,S.shape[1]):
            s[1:] = np.maximum(s[1:], np.maximum.accumulate(s[:-1]+S[:,i]))

    else:
        s = np.zeros(tuple([S.shape[1]+1])+S.shape[2:])

        for i in range(0,S.shape[0]):
            s[1:] = np.maximum(s[1:], np.maximum.accumulate(s[:-1]+S[i]))

    return s[-1]

def approximate_pairwise_matches_score(S):
    return S.clip(0).sum(axis=(0,1))

def sort_matches(S,matches):
    idx = np.argsort(S[[tuple(matches.T[1]),tuple(matches.T[0])]])[::-1]
    return matches[idx]


######################
# Statistics Functions
######################
def ordered_partition(n,m,rng):
    order = SortedSet(range(n))

    if m < n-m:
        choice = rng.choice(order, m, replace=False)
        return np.concatenate([order & choice,
                               order ^ choice])
    else:
        choice = rng.choice(order, n-m, replace=False)
        return np.concatenate([order ^ choice,
                               order & choice])

def null_distribution(S, mzs1, mzs2, pmz1=None, pmz2=None, kind='align',
                      iterations=1000, seed=None):
    if kind == 'align':
        score_func = fast_pairwise_alignment_score
    elif kind == 'match':
        score_func = approximate_pairwise_matches_score
    else:
        assert kind in ['align', 'match']

    rng = np.random.RandomState(seed)

    idx_perms = np.zeros((S.shape[0], iterations), dtype=int)
    for i in range(iterations):
        idx_perms[:,i] = ordered_partition(S.shape[0],len(mzs1),rng)

    mz_order = np.argsort(np.concatenate([mzs1,mzs2]))

    if pmz1 is not None and pmz2 is not None:
        nl_order = np.argsort(np.concatenate([pmz1-mzs1,pmz2-mzs2]))
        null_dist = score_func(S[mz_order[idx_perms][:len(mzs1), None],
                                 mz_order[idx_perms][len(mzs1):]]+
                               S[nl_order[idx_perms[:,::-1]][:len(mzs1), None],
                                 nl_order[idx_perms[:,::-1]][len(mzs1):]])/2
    else:
        null_dist = score_func(S[mz_order[idx_perms][:len(mzs1), None],
                                 mz_order[idx_perms][len(mzs1):]])

    return null_dist

def significance_test(S, mzs1, mzs2, pmz1=None, pmz2=None, kind='align',
                      max_log_iter=3, return_dist=False, early_stop=True, seed=None):
    assert isinstance(max_log_iter, int)

    if kind == 'align':
        observed_score = fast_pairwise_alignment_score(S[:len(mzs1),len(mzs1):])
    elif kind == 'match':
        observed_score = approximate_pairwise_matches_score(S[:len(mzs1),len(mzs1):])
    else:
        assert kind in ['align', 'match']

    null_dist = []
    p_value = 1.0

    if early_stop:
        start = 1
    else:
        start = max_log_iter

    for log_iter in range(start, max_log_iter+1):
        iterations = 10**log_iter

        null_dist.extend(null_distribution(S, mzs1, mzs2, pmz1, pmz2, kind,
                                           iterations-len(null_dist), seed))

        new_p_value = (observed_score <= np.array(null_dist)).sum()/iterations
        new_p_value = max(new_p_value, 1.0/iterations)

        if (new_p_value/p_value) <= .5:
            p_value = new_p_value
        else:
            p_value = new_p_value
            break

    if return_dist:
        return p_value, null_dist
    else:
        return p_value
