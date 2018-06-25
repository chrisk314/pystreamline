from unittest import TestCase

import numpy as np
from numpy import random as npr

from pystreamline import StreamlineIntegrator


class TestStreamlineIntegrator(TestCase):

    """Tests methods of wrapped c++ class PyStreamlineIntegrator."""

    def test_StreamlineIntegrator_ctor(self):
        """Tests that ``bounds`` attribute is set correctly."""
        pos = npr.random((10, 3))
        streamline_integrator = StreamlineIntegrator(pos)
        np.testing.assert_allclose(
            np.column_stack((np.min(pos, axis=0), np.max(pos, axis=0))),
            streamline_integrator.get_bounds()
        )

    def test_StreamlineIntegrator_get_cell_indices_in_range(self):
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
