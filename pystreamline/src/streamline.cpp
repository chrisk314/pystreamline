#include <cassert>
#include <cmath>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

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

static void delete_node_int_data(void* data)
{
    delete (int*) data;
}

template<typename TK, typename TV>
std::vector<TK> extract_keys_from_unordered_map(std::unordered_map<TK, TV> const& input_map)
{
    std::vector<TK> key_list;
    for (auto const& element : input_map) {
        key_list.push_back(element.first);
    }
    return key_list;
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


StreamlineIntegrator::~StreamlineIntegrator()
{
    tree->destr = delete_node_int_data;
    kd_free(tree);
}


int StreamlineIntegrator::get_n_points()
{
    return n_points;
}


int StreamlineIntegrator::get_dim()
{
    return dim;
}


double* StreamlineIntegrator::get_bounds()
{
    return bounds;
}


int StreamlineIntegrator::set_vec(std::string name, std::string *vec_name_ref,
  double **vec_ref, bool *vec_set_ref)
{
    if ( var_store_double.find(name) != var_store_double.end() ) {
        *vec_name_ref = name;
        *vec_ref = var_store_double[name];
        *vec_set_ref = true;
    }
    else {
        char message[100];
        sprintf(message, "No array with name: %s", name.c_str());
        throw std::invalid_argument(message);
    }
    return 0;
}


int StreamlineIntegrator::set_vec_x(std::string name)
{
    set_vec(name, &vec_x_name, &vec_x, &vec_x_set);
    return 0;
}


int StreamlineIntegrator::set_vec_y(std::string name)
{
    set_vec(name, &vec_y_name, &vec_y, &vec_y_set);
    return 0;
}


int StreamlineIntegrator::set_vec_z(std::string name)
{
    set_vec(name, &vec_z_name, &vec_z, &vec_z_set);
    return 0;
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


int StreamlineIntegrator::set_interp_lscale(double _interp_lscale)
{
    interp_lscale = _interp_lscale;
    neg_inv_interp_lscale_sq = -1. / (interp_lscale * interp_lscale);
    return 0;
}


int StreamlineIntegrator::interpolate_vec_at_point(double *pos, double *vec)
{
    int count, *idx;
    double *dist_sq;

    get_points_in_range(pos[0], pos[1], pos[2], interp_lscale, &count, &idx, &dist_sq);

    memset(vec, 0, 3*sizeof(double));

    double norm = 0.;
    for (int i=0; i<count; i++)
    {
        int tmp_idx = idx[i];
        double weight = exp(neg_inv_interp_lscale_sq * dist_sq[i]);
        norm += weight;
        vec[0] += vec_x[tmp_idx] * weight;
        vec[1] += vec_y[tmp_idx] * weight;
        vec[2] += vec_z[tmp_idx] * weight;
    }
    norm = count > 0 ? 1. / norm : 1.;
    for (int i=0; i<3; i++)
    {
        vec[i] *= norm;
    }

    delete [] idx;
    delete [] dist_sq;

    return 0;
}


int StreamlineIntegrator::add_int_array(std::string name, int *arr)
{
    if ( var_store_int.find(name) == var_store_int.end() )
        var_store_int.insert({name, arr});
    else
        var_store_int[name] = arr;
    return 0;
}


int* StreamlineIntegrator::get_int_array_with_name(std::string name)
{
    if ( var_store_int.find(name) == var_store_int.end() )
    {
        char message[100];
        sprintf(message, "key: %s not in int store", name.c_str());
        throw std::invalid_argument(message);
    }
    else
    {
        return var_store_int[name];
    }
}


std::vector<std::string> StreamlineIntegrator::get_int_array_names()
{
    return extract_keys_from_unordered_map(var_store_int);
}


int StreamlineIntegrator::add_double_array(std::string name, double *arr)
{
    if ( var_store_double.find(name) == var_store_double.end() )
        var_store_double.insert({name, arr});
    else
        var_store_double[name] = arr;
    return 0;
}


double* StreamlineIntegrator::get_double_array_with_name(std::string name)
{
    if ( var_store_double.find(name) == var_store_double.end() )
    {
        char message[100];
        sprintf(message, "key: %s not in double store", name.c_str());
        throw std::invalid_argument(message);
    }
    else
    {
        return var_store_double[name];
    }
}


std::vector<std::string> StreamlineIntegrator::get_double_array_names()
{
    return extract_keys_from_unordered_map(var_store_double);
}

}
