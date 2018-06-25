#include "kdtree.hpp"

namespace PyStreamline
{

class StreamlineIntegrator
{
    int n_points, dim;
    double bounds[6];
    struct kdtree *tree;

  public:
    StreamlineIntegrator(double*, int, int);
    int get_points_in_range(double, double, double, double, int*, int**, double**);
    double* get_bounds();
};

}
