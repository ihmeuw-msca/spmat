import numpy as np
cimport numpy as cnp
cimport cython
cimport scipy.linalg.cython_blas as cblas
cimport scipy.linalg.cython_lapack as clapack


def block_lsvd(double[:, ::1] a_view,
               long[::1] n_view,
               double[::1] u_view,
               double[::1] s_view):
    cdef int dim_row = max(n_view)
    cdef int dim_col = a_view.shape[1]
    cdef int dim_max = max(dim_row, dim_col)
    cdef int dim_min = min(dim_row, dim_col)
    cdef int lwork = 2*max(1, 3*dim_min + dim_max, 5*dim_min)
    cdef int info
    cdef int ind_a = 0
    cdef int ind_u = 0
    cdef int ind_s = 0

    w = np.empty(shape=(lwork,), dtype=np.float_)
    cdef double[::1] w_view = w


    for i in range(n_view.size):
        dim_row = n_view[i]
        dim_min = min(dim_row, dim_col)

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
