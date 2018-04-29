from scipy.sparse import isspmatrix, dok_matrix, csc_matrix, csr_matrix
import cython_sparse as cs
cimport cython_sparse as cs

cpdef cs.SparseMatrix add_self_loops_sm(cs.SparseMatrix sm, double loopval):
    """
    Creates a copy of sm with its diagonal set to sm.
    :param sm: 
    :param loopval: 
    :return: The resultant matrix.
    """

    cdef:
        cs.IncrementalSparseMatrix res_inc

        int i, j, jj, start, end
        Py_ssize_t iln = len(sm.indptr) - 1

    assert iln == sm.minor_shape, "Must be a square matrix."

    # Max buffer occurs if no diagonals are set.
    with cs.IncrementalSparseMatrix(iln, sm.minor_shape, sm.nnz + iln) as res_inc:

        for i in range(iln):
            start = sm.indptr[i]
            end = sm.indptr[i+1]
            for jj in range(start, end):
                j = sm.indices[jj]

                if i == j:
                    res_inc.set(i, j, loopval)
                else:
                    res_inc.set(i, j, sm.data[jj])

    return res_inc.sm

cdef long _prune_sm(cs.SparseMatrix sm, double threshold) except -1:
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
        Py_ssize_t M = len(sm.indptr) - 1
        long cursor = 0
        double val

    start = sm.indptr[0]
    for i in range(M):
        end = sm.indptr[i + 1]
        frsize = 0  # final row size, after pruning.

        for j in range(start, end):
            val = sm.data[j]
            if val >= threshold:
                sm.data[cursor] = val
                sm.indices[cursor] = sm.indices[j]
                cursor += 1
                frsize += 1

        # Update the end in the array.
        sm.indptr[i + 1] = sm.indptr[i] + frsize
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
    cdef cs.SparseMatrix sm

    if isspmatrix(matrix):
        # matrix = matrix.copy()  # not great, but it's happening below so we want to be safe!
        # # print()
        # # print("original indptr:", matrix.indptr)
        # # print()
        #
        # # Create a SparseMatrix for passing to prune.
        # sm.indices = matrix.indices
        # sm.indptr = matrix.indptr
        # sm.data = matrix.data
        sm = cs.SparseMatrix(matrix.data, matrix.indices, matrix.indptr, matrix.shape[1])
        sm = sm.copy()  # just for safety, not sure if necessary.

        sidx = _prune_sm(sm, threshold)

        if isinstance(matrix, csc_matrix):
            # prune happens in place, so we refer to matrix to piggy back off of the python memory management coming
            #  down the pike.
            # print("post data:", matrix.data[:sidx])
            # print("post indices:", matrix.indices[:sidx])
            # print("post indptr:", matrix.indptr)
            # pruned = csc_matrix((matrix.data[:sidx], matrix.indices[:sidx], matrix.indptr), shape=matrix.shape)
            pruned = cs.tocsc(sm)
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