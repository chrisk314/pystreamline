from unittest import TestCase

import numpy as np
from numpy import random as npr

from pystreamline import StreamlineIntegrator


class TestStreamlineIntegrator(TestCase):

    """Tests methods of wrapped c++ class PyStreamlineIntegrator."""

    def setUp(self):
        self.n_points, self.dim = 10, 3
        self.pos = npr.random((self.n_points, self.dim))
        self.streamline_integrator = StreamlineIntegrator(self.pos)

    def tearDown(self):
        del self.streamline_integrator
        del self.pos

    def test_StreamlineIntegrator_ctor(self):
        """Tests that ``bounds`` attribute is set correctly."""
        np.testing.assert_allclose(
            np.column_stack((np.min(self.pos, axis=0), np.max(self.pos, axis=0))),
            self.streamline_integrator.get_bounds()
        )

    def test_StreamlineIntegrator_var_store_int(self):
        # Tests ``arr1`` data can be set correctly
        arr1 = npr.randint(0, 100, self.n_points, dtype=np.int32)
        self.streamline_integrator.add_int_array('arr1', arr1)
        np.testing.assert_allclose(
            arr1, self.streamline_integrator.get_int_array_with_name('arr1')
        )

        # Tests ``arr1`` data can be overwritten correctly
        arr1 = npr.randint(0, 100, self.n_points, dtype=np.int32)
        self.streamline_integrator.add_int_array('arr1', arr1)
        np.testing.assert_allclose(
            arr1, self.streamline_integrator.get_int_array_with_name('arr1')
        )

        # Tests setting multiple data arrays at once
        arrs = [
            ('arr2', npr.randint(0, 100, self.n_points, dtype=np.int32)),
            ('arr3', npr.randint(0, 100, self.n_points, dtype=np.int32))
        ]
        self.streamline_integrator.add_int_arrays(arrs)
        np.testing.assert_allclose(
            arrs[0][1], self.streamline_integrator.get_int_array_with_name('arr2')
        )
        np.testing.assert_allclose(
            arrs[1][1], self.streamline_integrator.get_int_array_with_name('arr3')
        )

        # Tests array names are stored and recovered correctly
        assert set(('arr1', 'arr2', 'arr3')) == set(self.streamline_integrator.int_array_names)

        # Tests exception is raised when array name is not present
        with self.assertRaises(ValueError):
            self.streamline_integrator.get_int_array_with_name('arr4')

    def test_StreamlineIntegrator_var_store_double(self):
        # Tests ``arr1`` data can be set correctly
        arr1 = npr.random(self.n_points)
        self.streamline_integrator.add_double_array('arr1', arr1)
        np.testing.assert_allclose(
            arr1, self.streamline_integrator.get_double_array_with_name('arr1')
        )

        # Tests ``arr1`` data can be overwritten correctly
        arr1 = npr.random(self.n_points)
        self.streamline_integrator.add_double_array('arr1', arr1)
        np.testing.assert_allclose(
            arr1, self.streamline_integrator.get_double_array_with_name('arr1')
        )

        # TODO : running with this code causes segfault.
        # -->
        # Tests setting multiple data arrays at once
        arrs = [
            ('arr2', npr.random(self.n_points)),
            ('arr3', npr.random(self.n_points))
        ]
        self.streamline_integrator.add_double_arrays(arrs)
        np.testing.assert_allclose(
            arrs[0][1], self.streamline_integrator.get_double_array_with_name('arr2')
        )
        np.testing.assert_allclose(
            arrs[1][1], self.streamline_integrator.get_double_array_with_name('arr3')
        )
        # <--

        # Tests array names are stored and recovered correctly
        assert set(('arr1', 'arr2', 'arr3')) == set(self.streamline_integrator.double_array_names)

        # Tests exception is raised when array name is not present
        with self.assertRaises(ValueError):
            self.streamline_integrator.get_double_array_with_name('arr4')

    def test_StreamlineIntegrator_get_points_in_range(self):
        """Tests that kdtree query works correctly."""
        pos = np.array([
            [0.1, 0.1, 0.1],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.6],
        ])
        streamline_integrator = StreamlineIntegrator(pos)
        idx, dist_sq = streamline_integrator.get_points_in_range(.5, .5, .5, .1)

        assert len(idx) == 2
        np.testing.assert_allclose(np.array([0., 0.01]), dist_sq)
