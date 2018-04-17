from scipy.sparse import isspmatrix, dok_matrix, csc_matrix, csr_matrix

cdef struct SparseMatrix:
    double[::1] vals
    int[::1] indptrs
    int[::1] indices

cdef long _prune_sm(SparseMatrix sm, double threshold) except -1:
    """
    Take a sparse matrix and effectively remove values less than the threshold.
    This moves the valid elements up in the array, replacing the invalid elements.
    The number of valid elements after shifting is returned, and it is the caller's
     responsibility to slice away the excess accordingly.
    :param sm: 
    :param threshold: 
    :return:
        An idx such that vals[:idx] and indices[:idx] yields pruned value and index arrays. 
    """

    cdef:
        long i, j, start, end, frsize
        Py_ssize_t M = len(sm.indptrs) - 1
        long cursor = 0
        double val

    start = sm.indptrs[0]
    for i in range(M):
        end = sm.indptrs[i+1]
        frsize = 0  # final row size, after pruning.

        for j in range(start, end):
            val = sm.vals[j]
            if val >= threshold:
                sm.vals[cursor] = val
                sm.indices[cursor] = sm.indices[j]
                cursor += 1
                frsize += 1

        # Update the end in the array.
        sm.indptrs[i+1] = sm.indptrs[i] + frsize
        # Take the original end and set that to the next row's start.
        start = end

    return cursor + 1


cpdef prune(matrix, threshold):
    """
    Prune the matrix so that very small edges are removed
    
    :param matrix: The matrix to be pruned
    :param threshold: The value below which edges will be removed
    :returns: The pruned matrix
    """
    cdef SparseMatrix sm

    if isspmatrix(matrix):
        matrix = matrix.copy()  # not great, but it's happening below so we want to be safe!
        # print()
        # print("original indptr:", matrix.indptr)
        # print()

        # Create a SparseMatrix for passing to prune.
        sm.indices = matrix.indices
        sm.indptrs = matrix.indptr
        sm.vals = matrix.data

        sidx = _prune_sm(sm, threshold)

        if isinstance(matrix, csc_matrix):
            # prune happens in place, so we refer to matrix to piggy back off of the python memory management coming
            #  down the pike.
            # print("post data:", matrix.data[:sidx])
            # print("post indices:", matrix.indices[:sidx])
            # print("post indptr:", matrix.indptr)
            pruned = csc_matrix((matrix.data[:sidx], matrix.indices[:sidx], matrix.indptr), shape=matrix.shape)
        elif isinstance(matrix, csr_matrix):
            raise NotImplementedError()
        else:
            raise NotImplementedError()

        # pruned = dok_matrix(matrix.shape)
        # pruned[matrix >= threshold] = matrix[matrix >= threshold]
        # pruned = pruned.tocsc()
    else:
        pruned = matrix.copy()
        pruned[pruned < threshold] = 0

    return pruned