import numpy as np
cimport numpy as cnp
cimport cython
cimport scipy.linalg.cython_blas as blas
cimport scipy.linalg.cython_lapack as lapack


@cython.boundscheck(False)
@cython.wraparound(False)
def block_lsvd(double[:, ::1] a_view, long[::1] n_view, long[::1] k_view):
    # a_view: matrix, should be input data unrolled into 1D vector
    # n_view: block sizes
    # k_view: number of columns
    # need to rewrite block_lsvd to use 1D vectors

    # LSVD = left singular value decomposition

    # Expected input/output?
    # Based on comment, is this just SVD? Can we use np.linalg.svd instead?
    # What is "L" in "LSVD"? Assume "SVD" means singular value decomposition
    # So: numpy.linalg.svd uses Lapack.dgesdd instead of lapack.dgesvd, what's the difference? Comparable results?
    cdef int dim_row = max(n_view) # dimension of the rows
    cdef int dim_col = a_view.shape[1]
    cdef int dim_max = max(dim_row, dim_col) # dim_max = sz1
    cdef int dim_min = min(dim_row, dim_col) # dim_min = ss
    cdef int lwork = 2*max(1, 3*dim_min + dim_max, 5*dim_min)
    cdef int ind_a = 0
    cdef int ind_u = 0
    cdef int ind_s = 0
    cdef int info # Output of the svd. stored in &info

    u = np.empty(np.array(n_view).dot(np.array(k_view)), dtype=np.float_)
    s = np.empty(np.array(k_view).sum(), dtype=np.float_)
    w = np.empty(lwork, dtype=np.float_)
    cdef double[::1] u_view = u
    cdef double[::1] s_view = s
    cdef double[::1] w_view = w

    # What is input for n_view? Seems to be a way to chunk a matrix into block matrices and run SVD on each block?
    for i in range(n_view.size):
        dim_row = n_view[i]
        dim_min = k_view[i]
        # Might be worth checking:
        # https://netlib.org/lapack/lug/node71.html#:~:text=We%20include%20both%20DGESVD%20and,than%20DGESVD%20on%20most%20machines.
        # DGESDD supposed to be much faster
        lapack.dgesvd("N", "S",
                      &dim_col, &dim_row,
                      &a_view[ind_a][0], &dim_col,
                      &s_view[ind_s],
                      &a_view[ind_a][0], &dim_col,
                      &u_view[ind_u], &dim_min,
                      &w_view[0], &lwork, &info)

        ind_a += dim_row
        ind_u += dim_row*dim_min
        ind_s += dim_min

    return u, s


@cython.boundscheck(False)
@cython.wraparound(False)
def block_mvdot(double[::1] u_view,
                double[::1] v_view,
                double[::1] x_view,
                long[::1] n_view,
                long[::1] k_view):
    # Mi = Ui Sigma Vi(T)
    # U = U
    # V = V
    # x = term to be multiplied
    # k =
    cdef int dim_row
    cdef int dim_col
    cdef int one_int = 1
    cdef double zero_double = 0
    cdef double one_double = 1

    t = np.empty(np.array(k_view).sum(), dtype=np.float_)
    y = np.empty(np.array(n_view).sum(), dtype=np.float_)
    cdef double[::1] y_view = y
    cdef double[::1] t_view = t
    cdef int i
    cdef int ind_x = 0
    cdef int ind_y = 0
    cdef int ind_t = 0
    cdef int ind_u = 0

    # compute t = u.T @ x
    for i in range(n_view.size):
        dim_row = n_view[i]
        dim_col = k_view[i]

        blas.dgemv("N",
                   &dim_col, &dim_row, &one_double,
                   &u_view[ind_u], &dim_col,
                   &x_view[ind_x], &one_int, &zero_double,
                   &t_view[ind_t], &one_int)

        ind_x += dim_row
        ind_t += dim_col
        ind_u += dim_row*dim_col

    # compute t = t * v
    for i in range(t_view.size):
        t_view[i] *= v_view[i]

    # compute y = u @ t
    ind_t = 0
    ind_u = 0
    for i in range(n_view.size):
        dim_row = n_view[i]
        dim_col = k_view[i]

        blas.dgemv("T",
                   &dim_col, &dim_row, &one_double,
                   &u_view[ind_u], &dim_col,
                   &t_view[ind_t], &one_int, &zero_double,
                   &y_view[ind_y], &one_int)

        ind_y += dim_row
        ind_t += dim_col
        ind_u += dim_row*dim_col

    # compute y = y + x
    for i in range(y_view.size):
        y_view[i] += x_view[i]

    return y


@cython.boundscheck(False)
@cython.wraparound(False)
def block_mmdot(double[::1] u_view,
                double[::1] v_view,
                double[:, ::1] x_view,
                long[::1] n_view,
                long[::1] k_view):
    cdef int dim_row
    cdef int dim_col
    cdef int num_col = x_view.shape[1]
    cdef double zero_double = 0
    cdef double one_double = 1

    t = np.empty((np.array(k_view).sum(), num_col), dtype=np.float_)
    y = np.empty((np.array(n_view).sum(), num_col), dtype=np.float_)
    cdef double[:, ::1] y_view = y
    cdef double[:, ::1] t_view = t
    cdef int i, j
    cdef int ind_x = 0
    cdef int ind_y = 0
    cdef int ind_t = 0
    cdef int ind_u = 0

    # compute t = u.T @ x
    for i in range(n_view.size):
        dim_row = n_view[i]
        dim_col = k_view[i]
        
        blas.dgemm("N", "T",
                   &num_col, &dim_col, &dim_row, &one_double,
                   &x_view[ind_x][0], &num_col,
                   &u_view[ind_u], &dim_col, &zero_double,
                   &t_view[ind_t][0], &num_col)

        ind_x += dim_row
        ind_t += dim_col
        ind_u += dim_row*dim_col

    # compute t = t * v
    for i in range(t_view.shape[0]):
        for j in range(t_view.shape[1]):
            t_view[i, j] *= v_view[i]

    # compute y = u @ t
    ind_u = 0
    ind_t = 0
    for i in range(n_view.size):
        dim_row = n_view[i]
        dim_col = k_view[i]

        blas.dgemm("N", "N",
                   &num_col, &dim_row, &dim_col, &one_double,
                   &t_view[ind_t][0], &num_col,
                   &u_view[ind_u], &dim_col, &zero_double,
                   &y_view[ind_y][0], &num_col)

        ind_y += dim_row
        ind_t += dim_col
        ind_u += dim_row*dim_col

    # compute y = y + x
    for i in range(y_view.shape[0]):
        for j in range(y_view.shape[1]):
            y_view[i, j] += x_view[i, j]

    return y
