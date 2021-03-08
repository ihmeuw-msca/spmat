import numpy as np
cimport numpy as cnp
cimport cython
cimport scipy.linalg.cython_blas as cblas
cimport scipy.linalg.cython_lapack as clapack


def block_lsvd(double[:, ::1] a_view, long[::1] n_view, long[::1] k_view):
    cdef int dim_row = max(n_view)
    cdef int dim_col = a_view.shape[1]
    cdef int dim_max = max(dim_row, dim_col)
    cdef int dim_min = min(dim_row, dim_col)
    cdef int lwork = 2*max(1, 3*dim_min + dim_max, 5*dim_min)
    cdef int ind_a = 0
    cdef int ind_u = 0
    cdef int ind_s = 0
    cdef int info

    u = np.empty(np.array(n_view).dot(np.array(k_view)), dtype=np.float_)
    s = np.empty(np.array(k_view).sum(), dtype=np.float_)
    w = np.empty(lwork, dtype=np.float_)
    cdef double[::1] u_view = u
    cdef double[::1] s_view = s
    cdef double[::1] w_view = w


    for i in range(n_view.size):
        dim_row = n_view[i]
        dim_min = k_view[i]

        clapack.dgesvd("N", "S",
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


def block_mvdot(double[::1] u_view,
                double[::1] v_view,
                double[::1] x_view,
                long[::1] n_view,
                long[::1] k_view):
    cdef int dim_row
    cdef int dim_col
    cdef int one_int = 1
    cdef double zero_double = 0
    cdef double one_double = 1

    t = np.empty(np.array(k_view).max(), dtype=np.float_)
    y = np.empty(np.array(n_view).sum(), dtype=np.float_)
    cdef double[::1] y_view = y
    cdef double[::1] t_view = t
    cdef int ind_x = 0
    cdef int ind_y = 0
    cdef int ind_u = 0
    cdef int ind_v = 0

    for i in range(n_view.size):
        dim_row = n_view[i]
        dim_col = k_view[i]

        # compute t = u.T @ x
        cblas.dgemv("N",
                    &dim_col, &dim_row, &one_double,
                    &u_view[ind_u], &dim_col,
                    &x_view[ind_x], &one_int, &zero_double,
                    &t_view[0], &one_int)
        
        # compute t = t * v
        for j in range(dim_col):
            t_view[j] *= v_view[ind_v + j]
        
        # compute y = u @ t
        cblas.dgemv("T",
                    &dim_col, &dim_row, &one_double,
                    &u_view[ind_u], &dim_col,
                    &t_view[0], &one_int, &zero_double,
                    &y_view[ind_y], &one_int)

        # compute y = y + x
        for j in range(dim_row):
            y_view[ind_y + j] += x_view[ind_x + j]

        # update indices
        ind_x += dim_row
        ind_y += dim_row
        ind_u += dim_row*dim_col
        ind_v += dim_col

    return y


def block_mmdot(double[::1] u_view,
                double[::1] v_view,
                double[:, ::1] x_view,
                long[::1] n_view,
                long[::1] k_view):
    cdef int dim_u_row
    cdef int dim_u_col
    cdef int dim_x_col = x_view.shape[1]
    cdef double zero_double = 0
    cdef double one_double = 1

    t = np.empty((np.array(k_view).max(), dim_x_col), dtype=np.float_)
    y = np.empty((np.array(n_view).sum(), dim_x_col), dtype=np.float_)
    cdef double[:, ::1] y_view = y
    cdef double[:, ::1] t_view = t
    cdef int ind_x = 0
    cdef int ind_y = 0
    cdef int ind_u = 0
    cdef int ind_v = 0

    for i in range(n_view.size):
        dim_u_row = n_view[i]
        dim_u_col = k_view[i]

        # compute t = u.T @ x
        cblas.dgemm("N", "T",
                    &dim_x_col, &dim_u_row, &dim_u_row, &one_double,
                    &x_view[ind_x][0], &dim_x_col,
                    &u_view[ind_u], &dim_u_col, &zero_double,
                    &t_view[0][0], &dim_x_col)
        
        # compute t = t * v
        for j in range(dim_u_col):
            for k in range(dim_x_col):
                t_view[j][k] *= v_view[ind_v + j]
        
        # compute y = u @ t
        cblas.dgemm("N", "N",
                    &dim_x_col, &dim_u_col, &dim_u_col, &one_double,
                    &t_view[0][0], &dim_x_col,
                    &u_view[ind_u], &dim_u_col, &zero_double,
                    &y_view[ind_y][0], &dim_x_col)

        # compute y = y + x
        for j in range(dim_u_row):
            for k in range(dim_x_col):
                y_view[ind_y + j][k] += x_view[ind_x + j][k]

        # update indices
        ind_x += dim_u_row
        ind_y += dim_u_row
        ind_u += dim_u_row*dim_u_col
        ind_v += dim_u_col

    return y