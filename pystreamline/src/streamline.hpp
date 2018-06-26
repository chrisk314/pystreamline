#include <unordered_map>
#include <string>
#include <vector>

#include "kdtree.hpp"

namespace PyStreamline
{

class StreamlineIntegrator
{
    int n_points, dim;
    double bounds[6];
    struct kdtree *tree;

    std::unordered_map<std::string, double*> var_store_double;

    double interp_lscale = 0., neg_inv_interp_lscale_sq = 0.;
    double *vec_x, *vec_y, *vec_z;
    bool vec_x_set = false, vec_y_set = false, vec_z_set = false;
    std::string vec_x_name, vec_y_name, vec_z_name;

    int set_vec(std::string, std::string*, double**, bool*);

  public:
    StreamlineIntegrator(double*, int, int);
    ~StreamlineIntegrator();

    int get_n_points();
    int get_dim();
    double* get_bounds();

    int set_vec_x(std::string);
    int set_vec_y(std::string);
    int set_vec_z(std::string);
    int set_interp_lscale(double);

    int get_points_in_range(double, double, double, double, int*, int**, double**);
    int interpolate_vec_at_point(double*, double*);

    int add_double_array(std::string, double*);
    std::vector<std::string> get_double_array_names();
    double* get_double_array_with_name(std::string);
};

}
