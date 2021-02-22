import numpy as np
from scipy.linalg import eig
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


###############################
# Similarity Matrix Functions
###############################
def counts_matrix(A, tolerance=.01):
    counts = np.zeros_like(A, dtype=float)

    sorted_elements = SortedList(np.ndenumerate(A), key=lambda i:i[1])

    for (i,j),v in sorted_elements:
        left =  ((None,None), v - (tolerance/2))
        right = ((None,None), v + (tolerance/2))

        counts[i,j] = sorted_elements.bisect_right(right) - \
                      sorted_elements.bisect_left(left)

    return counts

def substitution_matrix(mzs1, mzs2, tolerance=.01):
    mzs = np.concatenate([mzs1,mzs2])

    mz_diffs = np.subtract.outer(mzs,mzs)

    frag_count = np.zeros_like(mz_diffs)

    frag_count[:len(mzs1)] += counts_matrix(mz_diffs[:len(mzs1)], tolerance)
    frag_count[len(mzs1):] += counts_matrix(mz_diffs[len(mzs1):], tolerance)

    frag_sub = np.linalg.pinv(laplacian(frag_count))

    return frag_sub


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

def fast_pairwise_align(S):
    if S.shape[0] < S.shape[1]:
        s = np.zeros(tuple([S.shape[0]+1])+S.shape[2:])

        for i in range(0,S.shape[1]):
            s[1:] = np.maximum(s[1:], np.maximum.accumulate(s[:-1]+S[:,i]))

    else:
        s = np.zeros(tuple([S.shape[1]+1])+S.shape[2:])

        for i in range(0,S.shape[0]):
            s[1:] = np.maximum(s[1:], np.maximum.accumulate(s[:-1]+S[i]))

    return s[-1]


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

def null_distribution(S, mzs1, mzs2, iterations=1000, seed=None):
    mz_order = np.argsort(np.concatenate([mzs1,mzs2]))

    rng = np.random.RandomState(seed)

    S1_perms = np.zeros((len(mzs1), S.shape[0]-len(mzs1), iterations))
    S2_perms = np.zeros((S.shape[0]-len(mzs1), len(mzs1), iterations))

    for i in range(iterations):
        idx_order = ordered_partition(S.shape[0],len(mzs1),rng)
        mz_idx = mz_order[idx_order]

        S1_perms[:,:,i] = S[mz_idx[:len(mzs1),None],mz_idx[len(mzs1):]]
        S2_perms[:,:,i] = S[mz_idx[len(mzs1):,None],mz_idx[:len(mzs1)]]

    null_dist = (fast_pairwise_align(S1_perms)+
                 fast_pairwise_align(S2_perms))/2

    return null_dist

def alignment_test(S, mzs1, mzs2, max_log_iter=3, early_stop=True, seed=None):
    assert isinstance(max_log_iter, int)

    observed_score = (fast_pairwise_align(S[:len(mzs1),len(mzs1):])+
                      fast_pairwise_align(S[len(mzs1):,:len(mzs1)]))/2

    null_dist = []
    p_value = 1.0

    if early_stop:
        start = 1
    else:
        start = max_log_iter

    for log_iter in range(start, max_log_iter+1):
        iterations = 10**log_iter

        null_dist.extend(null_distribution(S, mzs1, mzs2,
                                           iterations-len(null_dist), seed))

        new_p_value = (observed_score <= np.array(null_dist)).sum()/iterations
        new_p_value = max(new_p_value, 1.0/iterations)

        if (new_p_value/p_value) <= .5:
            p_value = new_p_value
        else:
            p_value = new_p_value
            break

    return p_value
