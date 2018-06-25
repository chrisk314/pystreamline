#include <cassert>

#include "kdtree.hpp"
#include "streamline.hpp"

static double calc_dist_sq( double *a1, double *a2, int dims ) {
  double dist_sq = 0, diff;
  while( --dims >= 0 ) {
    diff = (a1[dims] - a2[dims]);
    dist_sq += diff*diff;
  }
  return dist_sq;
}

namespace PyStreamline
{

StreamlineIntegrator::StreamlineIntegrator(double *pos, int _n_points, int _dim) :
  n_points(_n_points), dim(_dim)
{
    assert(_dim == 2 || _dim == 3);

    tree = kd_create(_dim);

    bounds[0] = bounds[1] = pos[0];
    bounds[2] = bounds[3] = pos[1];
    bounds[4] = bounds[5] = pos[2];

    for (int i=0; i<_n_points; i++)
    {
        int *idx = new int;
        *idx = i;
        int rval = kd_insert(tree, &pos[_dim*i], (void*) idx);
        assert(rval == 0);  // TODO : assert not working in Cython.

        if (pos[_dim*i] < bounds[0])
          bounds[0] = pos[_dim*i];
        if (pos[_dim*i] > bounds[1])
          bounds[1] = pos[_dim*i];
        if (pos[_dim*i+1] < bounds[2])
          bounds[2] = pos[_dim*i+1];
        if (pos[_dim*i+1] > bounds[3])
          bounds[3] = pos[_dim*i+1];
        if (pos[_dim*i+2] < bounds[4])
          bounds[4] = pos[_dim*i+2];
        if (pos[_dim*i+2] > bounds[5])
          bounds[5] = pos[_dim*i+2];
    }
}


double * StreamlineIntegrator::get_bounds()
{
    return bounds;
}


int StreamlineIntegrator::get_points_in_range(double x, double y, double z,
   double range, int* count, int** idx, double** dist_sq)
{
    struct kdres *rset = kd_nearest_range3(tree, x, y, z, range);

    *idx = new int[rset->size];
    *dist_sq = new double[rset->size];
    *count = 0;
    double pos[3];
    double pt[3] = {x, y, z};

    while (!kd_res_end(rset))
    {
        (*idx)[(*count)] = *((int*) kd_res_item(rset, pos));
        (*dist_sq)[(*count)] = calc_dist_sq( pt, pos, dim );
        (*count)++;
        kd_res_next(rset);
    }
    kd_res_free(rset);

    return 0;
}

}
