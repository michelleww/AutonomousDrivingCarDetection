%%cython --compile-args=-fopenmp --link-args=-fopenmp

from libc.math cimport sin, cos, pi, ceil, floor, pow
from libc.stdlib cimport abort, malloc, free
import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange, parallel
cimport openmp


cdef double get_pixel2d(
    double *image,
    Py_ssize_t n_rows, 
    Py_ssize_t n_cols,
    long x,
    long y) nogil:
    
    if (y < 0) or (y >= n_rows) or (x < 0) or (x >= n_cols):
        return 0
    else:
        return image[y * n_cols + x]

    
cdef double bilinear_interpolation(
    double *image,
    Py_ssize_t n_rows,
    Py_ssize_t n_cols,
    double x,
    double y) nogil:
    
    cdef double d_y, d_x, top_left, top_right, bottom_left, bottom_right
    cdef long min_y, min_x, max_y, max_x

    min_y = <long>floor(y)
    min_x = <long>floor(x)
    max_y = <long>ceil(y)
    max_x = <long>ceil(x)
    
    d_y = y - min_y
    d_x = x - min_x
    
    top_left = get_pixel2d(image, n_rows, n_cols, min_x, min_y)
    top_right = get_pixel2d(image, n_rows, n_cols, max_x, min_y)
    bottom_left = get_pixel2d(image, n_rows, n_cols, min_x, max_y)
    bottom_right = get_pixel2d(image, n_rows, n_cols, max_x, max_y)
    
    top = (1 - d_x) * top_left + d_x * top_right
    bottom = (1 - d_x) * bottom_left + d_x * bottom_right

    return (1 - d_y) * top + d_y * bottom


cdef double *joint_difference_distribution(
    double *image,
    Py_ssize_t n_rows,
    Py_ssize_t n_cols,
    int x0,
    int y0,
    int P,
    int R
) nogil:
    cdef Py_ssize_t p
    cdef double *T = <double *> malloc(sizeof(double) * P)
    cdef double x, y, gp, gc
    
    if T is NULL:
        abort()
        
    gc = get_pixel2d(image, n_rows, n_cols, x0, y0)
    
    for p in range(P):
        x = x0 + R * cos(2 * pi * p / P)
        y = y0 - R * sin(2 * pi * p / P)
        gp = bilinear_interpolation(image, n_rows, n_cols, x, y)
        T[p] = gp - gc
    
    return T


cdef int *binary_joint_distribution(double *T, Py_ssize_t T_size) nogil:
    cdef int *s_T = <int *> malloc(sizeof(int) * T_size)
    cdef Py_ssize_t i = 0
    
    for t in range(T_size):
        if T[t] >= 0.0:
            s_T[t] = 1
        else:
            s_T[t] = 0
    
    return s_T


cdef long LBP(double *T, int *s_T, Py_ssize_t T_size) nogil:
    cdef long LBP_pr = 0
    cdef Py_ssize_t i = 0
    
    for i in range(0, T_size):
        LBP_pr = LBP_pr + 2 ** i * s_T[i]
        
    return LBP_pr


cdef int is_uniform_pattern(int *s_T, Py_ssize_t s_T_size) nogil:
    cdef Py_ssize_t i = 0
    cdef int counter = 0
    
    for i in range(s_T_size - 1):
        if s_T[i] != s_T[i + 1]:
            counter += 1
        
        if counter > 2:
            return 0
    return 1


cdef int create_index(int *s_T, Py_ssize_t s_T_size) nogil:
    cdef int n_ones = 0
    cdef int rot_index = -1
    cdef int first_one = -1
    cdef int first_zero = -1
    cdef int lbp = -1

    cdef Py_ssize_t i
    for i in range(s_T_size):
        if s_T[i]:
            n_ones += 1
            if first_one == -1:
                first_one = i
        else:
            if first_zero == -1:
                first_zero = i
    
    if n_ones == 0:
        lbp = 0
    elif n_ones == s_T_size:
        lbp = s_T_size * (s_T_size - 1) + 1
    else:
        if first_one == 0:
            rot_index = n_ones - first_zero
        else:
            rot_index = s_T_size - first_one
        lbp = 1 + (n_ones - 1) * s_T_size + rot_index
    return lbp


cdef int LBP_uniform(int *s_T, Py_ssize_t s_T_size) nogil:
    cdef int LBP_pru = 0
    cdef Py_ssize_t i = 0
    
    if is_uniform_pattern(s_T, s_T_size):
        LBP_pru = create_index(s_T, s_T_size)
    else:
        LBP_pru = 2 + s_T_size * (s_T_size - 1)
        
    return LBP_pru


@cython.boundscheck(False)
@cython.wraparound(False)
def local_binary_patterns(
    double[:, ::1] image,
    int P,
    int R,
    int num_threads=1
):
    
    cdef Py_ssize_t x = 0
    cdef Py_ssize_t y = 0
    cdef int n_rows = image.shape[0]
    cdef int n_cols = image.shape[1]
    cdef int[:, ::1] lbp = np.zeros([n_rows, n_cols], dtype=np.int32) 
    
    with nogil, parallel(num_threads=num_threads):
        for y in prange(n_rows, schedule='static'):
            for x in prange(n_cols, schedule='static'):
                T = joint_difference_distribution(&image[0][0], n_rows, n_cols, x, y, P, R)
                s_T = binary_joint_distribution(T, P)
                lbp[y, x] = LBP_uniform(s_T, P)
    
    return np.asarray(lbp)