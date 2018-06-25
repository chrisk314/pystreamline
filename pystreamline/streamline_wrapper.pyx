# distutils: language = c++
# distutils: sources = pystreamline/src/streamline.cpp

import cython
cimport cython

import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

np.import_array()


cdef extern from "numpy/arrayobject.h":
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)


cdef data_to_numpy_int_array_with_spec(void * ptr, np.npy_intp N):
    cdef np.ndarray[np.int32_t, ndim=1] arr = \
        np.PyArray_SimpleNewFromData(1, &N, np.NPY_INT32, ptr)
    PyArray_ENABLEFLAGS(arr, np.NPY_OWNDATA)
    return arr


cdef data_to_numpy_double_array_with_spec(void * ptr, np.npy_intp N):
    cdef np.ndarray[np.float64_t, ndim=1] arr = \
        np.PyArray_SimpleNewFromData(1, &N, np.NPY_FLOAT64, ptr)
    PyArray_ENABLEFLAGS(arr, np.NPY_OWNDATA)
    return arr


cdef extern from "src/streamline.hpp" namespace "PyStreamline":
    cdef cppclass StreamlineIntegrator:
        int n_points
        int dim
        double bounds[6]
        StreamlineIntegrator(double*, int, int)
        int get_points_in_range(double, double, double, double, int*, int**, double**)
        double* get_bounds()


cdef class _StreamlineIntegrator__wrapper:

    cdef StreamlineIntegrator *thisptr

    def __cinit__(self, np.ndarray[np.float64_t, ndim=2, mode='c'] pos):
        pos = np.ascontiguousarray(pos, dtype=np.double)
        self.thisptr = new StreamlineIntegrator(&pos[0,0], pos.shape[0], pos.shape[1])

    def __dealloc__(self):
        del self.thisptr

    def get_points_in_range(self, double x, double y, double z, double range):
        cdef double *c_dist_sq_ptr
        cdef int *c_idx_ptr
        cdef int c_count
        self.thisptr.get_points_in_range(
            x, y, z, range, &c_count, &c_idx_ptr, &c_dist_sq_ptr
        )
        return (
            data_to_numpy_int_array_with_spec(c_idx_ptr, c_count),
            data_to_numpy_double_array_with_spec(c_dist_sq_ptr, c_count)
        )

    def get_bounds(self):
        cdef np.float64_t[:] view = <np.float64_t[:6]> self.thisptr.get_bounds()
        return np.asarray(view).reshape((3,2))
