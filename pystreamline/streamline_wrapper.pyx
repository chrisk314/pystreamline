# distutils: language = c++
# distutils: sources = pystreamline/src/streamline.cpp

import cython
cimport cython

import numpy as np
cimport numpy as np

from libcpp.vector cimport vector
from libcpp.string cimport string

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
        StreamlineIntegrator(double*, int, int)
        int get_n_points();
        int get_dim();
        int set_vec_x(string);
        int set_vec_y(string);
        int set_vec_z(string);
        int set_interp_lscale(double);
        int get_points_in_range(double, double, double, double, int*, int**, double**);
        int interpolate_vec_at_point(double*, double*);
        int add_int_array(string, int*);
        int* get_int_array_with_name(string);
        vector[string] get_int_array_names();
        int add_double_array(string, double*);
        double* get_double_array_with_name(string);
        vector[string] get_double_array_names();
        double* get_bounds()


cdef class _StreamlineIntegrator__wrapper:

    cdef StreamlineIntegrator *thisptr

    def __cinit__(self, np.ndarray[np.float64_t, ndim=2, mode='c'] pos):
        pos = np.ascontiguousarray(pos, dtype=np.double)
        self.thisptr = new StreamlineIntegrator(&pos[0,0], pos.shape[0], pos.shape[1])

    def __dealloc__(self):
        del self.thisptr

    @property
    def n_points(self):
        return self.thisptr.get_n_points();

    @property
    def dim(self):
        return self.thisptr.get_dim();

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

    def add_int_array(self, name, np.ndarray[np.int32_t, ndim=1, mode='c'] arr):
        cdef c_name = <string> name.encode('utf-8')
        arr = np.ascontiguousarray(arr, dtype=np.int32)
        self.thisptr.add_int_array(c_name, <int*> arr.data)

    def add_int_arrays(self, arr_list):
        for name, arr in arr_list:
            self.add_int_array(name, arr)

    @property
    def int_array_names(self):
        c_str_vector = self.thisptr.get_int_array_names()
        return [b.decode('utf-8') for b in c_str_vector]

    def get_int_array_with_name(self, name):
        try:
            assert name in self.int_array_names
        except AssertionError:
            raise ValueError('No int array with name: {:s}'.format(name))
        cdef string c_name = <string> name.encode('utf-8')
        cdef int* c_arr_ptr = <int*> self.thisptr.get_int_array_with_name(c_name)
        return data_to_numpy_int_array_with_spec(c_arr_ptr, self.n_points)

    def add_double_array(self, name, np.ndarray[np.float64_t, ndim=1, mode='c'] arr):
        cdef c_name = <string> name.encode('utf-8')
        arr = np.ascontiguousarray(arr, dtype=np.double)
        self.thisptr.add_double_array(c_name, <double*> arr.data)

    def add_double_arrays(self, arr_list):
        for name, arr in arr_list:
            self.add_double_array(name, arr)

    @property
    def double_array_names(self):
        c_str_vector = self.thisptr.get_double_array_names()
        return [b.decode('utf-8') for b in c_str_vector]

    def get_double_array_with_name(self, name):
        try:
            assert name in self.double_array_names
        except AssertionError:
            raise ValueError('No double array with name: {:s}'.format(name))
        cdef string c_name = <string> name.encode('utf-8')
        cdef double* c_arr_ptr = <double*> self.thisptr.get_double_array_with_name(c_name)
        return data_to_numpy_double_array_with_spec(c_arr_ptr, self.n_points)

    def set_vec_x(self, name):
        try:
            assert name in self.double_array_names
        except AssertionError:
            raise ValueError('No double array with name: {:s}'.format(name))
        cdef string c_name = <string> name.encode('utf-8')
        self.thisptr.set_vec_x(c_name)

    def set_vec_y(self, name):
        try:
            assert name in self.double_array_names
        except AssertionError:
            raise ValueError('No double array with name: {:s}'.format(name))
        cdef string c_name = <string> name.encode('utf-8')
        self.thisptr.set_vec_y(c_name)

    def set_vec_z(self, name):
        try:
            assert name in self.double_array_names
        except AssertionError:
            raise ValueError('No double array with name: {:s}'.format(name))
        cdef string c_name = <string> name.encode('utf-8')
        self.thisptr.set_vec_z(c_name)

    def set_vec_names(self, vec_x_name, vec_y_name, vec_z_name):
        self.set_vec_x_name(vec_x_name)
        self.set_vec_y_name(vec_y_name)
        self.set_vec_z_name(vec_z_name)

    def set_interp_lscale(self, lscale):
        self.thisptr.set_interp_lscale(<double> lscale)

    def _interp_vec_at_point(self, np.ndarray[np.float64_t, ndim=1, mode='c'] point):
        point = np.ascontiguousarray(point, dtype=np.double)
        cdef double c_vec[3]
        self.thisptr.interpolate_vec_at_point(<double*> point.data, c_vec)
        return np.array([x for x in c_vec])
