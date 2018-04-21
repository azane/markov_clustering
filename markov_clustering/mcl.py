import numpy as np
from scipy.sparse import isspmatrix, csc_matrix, csr_matrix
import sklearn.preprocessing
from .utils import MessagePrinter
import markov_clustering.c_mcl as c_mcl
import cython_sparse as cs


def normalize(matrix):

    if isspmatrix(matrix):
        assert type(matrix) == csc_matrix, "only csc format supported."

        matrix_c = matrix.copy()  # just for safety, not sure if necessary.
        sm = cs.SparseMatrix_factory(matrix_c.data, matrix_c.indices, matrix_c.indptr,
                                     matrix_c.shape[0], matrix_c.shape[1])
        cs.normalize_major_axis(sm)
        return cs.tocsc(sm)

    return sklearn.preprocessing.normalize(matrix)


def inflate(matrix, power):
    """
    Apply cluster inflation to the given matrix by raising
    each element to the given power.
    
    :param matrix: The matrix to be inflated
    :param power: Cluster inflation parameter
    :returns: The inflated matrix
    """

    if isspmatrix(matrix):
        assert type(matrix) == csc_matrix, "only csc format supported"

        sm = cs.SparseMatrix_factory(matrix.data, matrix.indices, matrix.indptr,
                                     matrix.shape[0], matrix.shape[1])
        sm = cs.copy(sm)  # just to be safe, not sure if necessary.

        # return normalize(matrix.power(power))
        cs.inflate_sparse(sm, float(power))
        cs.normalize_major_axis(sm)

        return cs.tocsc(sm)

    return sklearn.preprocessing.normalize(np.power(matrix, power))


def expand(matrix, power):
    """
    Apply cluster expansion to the given matrix by raising
    the matrix to the given power.
    
    :param matrix: The matrix to be expanded
    :param power: Cluster expansion parameter
    :returns: The expanded matrix
    """
    if isspmatrix(matrix):
        return matrix ** power

    return np.linalg.matrix_power(matrix, power)


def add_self_loops(matrix, loop_value):
    """
    Add self-loops to the matrix by setting the diagonal
    to loop_value
    
    :param matrix: The matrix to add loops to
    :param loop_value: Value to use for self-loops
    :returns: The matrix with self-loops
    """
    shape = matrix.shape
    assert shape[0] == shape[1], "Error, matrix is not square"

    if isspmatrix(matrix):
        new_matrix = matrix.todok()
    else:
        new_matrix = matrix.copy()

    for i in range(shape[0]):
        new_matrix[i, i] = loop_value

    if isspmatrix(matrix):
        return new_matrix.tocsc()

    return new_matrix


# Moved to c_mcl
# def prune(matrix, threshold)


def converged(matrix1, matrix2):
    """
    Check for convergence by determining if 
    matrix1 and matrix2 are approximately equal.
    
    :param matrix1: The matrix to compare with matrix2
    :param matrix2: The matrix to compare with matrix1
    :returns: True if matrix1 and matrix2 approximately equal
    """
    if isspmatrix(matrix1) or isspmatrix(matrix2):
        t1 = type(matrix1)
        if t1 != csr_matrix and t1 != csc_matrix:
            raise NotImplementedError("Non compressed matrices not supported.")
        if t1 != type(matrix2):
            raise NotImplementedError("Mismatched matrix types not supported.")

        sm1 = cs.SparseMatrix_factory(matrix1.data, matrix1.indices, matrix1.indptr,
                                      matrix1.shape[0], matrix1.shape[1])
        sm2 = cs.SparseMatrix_factory(matrix2.data, matrix2.indices, matrix2.indptr,
                                      matrix2.shape[0], matrix2.shape[1])
        return cs.sparse_all_close(sm1, sm2)

    return np.allclose(matrix1, matrix2)


def iterate(matrix, expansion, inflation):
    """
    Run a single iteration (expansion + inflation) of the mcl algorithm
    
    :param matrix: The matrix to perform the iteration on
    :param expansion: Cluster expansion factor
    :param inflation: Cluster inflation factor
    """
    # Expansion
    matrix = expand(matrix, expansion)

    # Inflation
    matrix = inflate(matrix, inflation)

    return matrix


def get_clusters(matrix):
    """
    Retrieve the clusters from the matrix
    
    :param matrix: The matrix produced by the MCL algorithm
    :returns: A list of tuples where each tuple represents a cluster and
              contains the indices of the nodes belonging to the cluster
    """
    if not isspmatrix(matrix):
        # cast to sparse so that we don't need to handle different 
        # matrix types
        matrix = csc_matrix(matrix)

    # get the attractors - non-zero elements of the matrix diagonal
    attractors = matrix.diagonal().nonzero()[0]

    # somewhere to put the clusters
    clusters = set()

    # the nodes in the same row as each attractor form a cluster
    for attractor in attractors:
        cluster = tuple(matrix.getrow(attractor).nonzero()[1].tolist())
        clusters.add(cluster)

    return sorted(list(clusters))


def run_mcl(matrix, expansion=2, inflation=2, loop_value=1,
            iterations=100, pruning_threshold=0.001, pruning_frequency=1,
            convergence_check_frequency=1, verbose=False):
    """
    Perform MCL on the given similarity matrix
    
    :param matrix: The similarity matrix to cluster
    :param expansion: The cluster expansion factor
    :param inflation: The cluster inflation factor
    :param loop_value: Initialization value for self-loops
    :param iterations: Maximum number of iterations
           (actual number of iterations will be less if convergence is reached)
    :param pruning_threshold: Threshold below which matrix elements will be set
           set to 0
    :param pruning_frequency: Perform pruning every 'pruning_frequency'
           iterations. 
    :param convergence_check_frequency: Perform the check for convergence
           every convergence_check_frequency iterations
    :param verbose: Print extra information to the console
    :returns: The final matrix
    """
    assert expansion > 1, "Invalid expansion parameter"
    assert inflation > 1, "Invalid inflation parameter"
    assert loop_value >= 0, "Invalid loop_value"
    assert iterations > 0, "Invalid number of iterations"
    assert pruning_threshold >= 0, "Invalid pruning_threshold"
    assert pruning_frequency > 0, "Invalid pruning_frequency"
    assert convergence_check_frequency > 0, "Invalid convergence_check_frequency"

    printer = MessagePrinter(verbose)

    printer.print("-" * 50)
    printer.print("MCL Parameters")
    printer.print("Expansion: {}".format(expansion))
    printer.print("Inflation: {}".format(inflation))
    if pruning_threshold > 0:
        printer.print("Pruning threshold: {}, frequency: {} iteration{}".format(
            pruning_threshold, pruning_frequency, "s" if pruning_frequency > 1 else ""))
    else:
        printer.print("No pruning")
    printer.print("Convergence check: {} iteration{}".format(
        convergence_check_frequency, "s" if convergence_check_frequency > 1 else ""))
    printer.print("Maximum iterations: {}".format(iterations))
    printer.print("{} matrix mode".format("Sparse" if isspmatrix(matrix) else "Dense"))
    printer.print("-" * 50)

    # Initialize self-loops
    if loop_value > 0:
        matrix = add_self_loops(matrix, loop_value)

    # Normalize
    matrix = normalize(matrix)

    # iterations
    for i in range(iterations):
        printer.print("Iteration {}".format(i + 1))

        # store current matrix for convergence checking
        last_mat = matrix.copy()

        # perform MCL expansion and inflation
        matrix = iterate(matrix, expansion, inflation)

        # prune
        if pruning_threshold > 0 and i % pruning_frequency == pruning_frequency - 1:
            printer.print("Pruning")
            matrix = c_mcl.prune(matrix, pruning_threshold)

        # Check for convergence
        if i % convergence_check_frequency == convergence_check_frequency - 1:
            printer.print("Checking for convergence")
            if converged(matrix, last_mat):
                printer.print("Converged after {} iteration{}".format(i + 1, "s" if i > 0 else ""))
                break

    print("MCL iters:", i)

    printer.print("-" * 50)

    return matrix
