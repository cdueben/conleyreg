#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#ifdef _OPENMP
  #include <omp.h>
#else
  #define omp_get_num_threads()  1
  #define omp_get_thread_num()   0
  #define omp_get_max_threads()  1
  #define omp_get_thread_limit() 1
  #define omp_get_num_procs()    1
#endif
// [[Rcpp::plugins(openmp)]]
#include <cmath>
#include <vector>
#include <cstddef>
#include "distance_functions.h"
#include "distance_matrices.h"
#include "XeeXhC.h"

// Function overview
// 1 ols Function computing dense distance matrix with a bartlett kernel and calculating spatial sandwich in ols case
// 2 ols_d_b Function computing sparse distance matrix with a bartlett kernel and batch insert and calculating spatial sandwich in ols case
// 3 ols_f_b Function computing sparse distance matrix with a bartlett kernel and batch insert using floats and calculating spatial sandwich in ols case
// 4 ols_d_b_p Function parallely computing sparse distance matrix with a bartlett kernel and batch insert and calculating spatial sandwich in ols case
// 5 ols_f_b_p Function parallely computing sparse distance matrix with a bartlett kernel and batch insert using floats and calculating spatial sandwich in ols case
// 6 ols_s_b Function computing sparse distance matrix with a uniform kernel and batch insert and calculating spatial sandwich in ols case
// 7 ols_s_b_p Function parallely computing sparse distance matrix with a uniform kernel and batch insert and calculating spatial sandwich in ols case
// 8 ols_r Function computing row-wise distances and calculating spatial sandwich in ols case

// 1 Function computing dense distance matrix with a bartlett kernel and calculating spatial sandwich in ols case
// [[Rcpp::export]]
arma::mat ols(arma::mat &coords, unsigned int n_obs, unsigned int n_obs_t, double dist_cutoff, arma::mat &X, arma::vec &e, unsigned int n_vars, bool haversine,
  bool sparse, bool bartlett, bool flt, unsigned int n_cores) {
  if(sparse) {
    if(bartlett) {
      if(flt) {
        // Bartlett, sparse, float
        arma::sp_fmat distances(n_obs_t, n_obs_t);
        dist_spmat_f(distances, coords, n_obs_t, dist_cutoff, haversine, n_cores);
        arma::mat XeeXh = XeeXhC_s_f(distances, X, e, n_obs, n_obs_t, n_vars, n_cores);
        return XeeXh;
      } else {
        // Bartlett, sparse, double
        arma::sp_mat distances(n_obs_t, n_obs_t);
        dist_spmat_d(distances, coords, n_obs_t, dist_cutoff, haversine, n_cores);
        arma::mat XeeXh = XeeXhC_s_d(distances, X, e, n_obs, n_obs_t, n_vars, n_cores);
        return XeeXh;
      }
    } else {
      // Uniform, sparse
      arma::SpMat<short int> distances(n_obs_t, n_obs_t);
      dist_spmat_s(distances, coords, n_obs_t, dist_cutoff, haversine, n_cores);
      arma::mat XeeXh = XeeXhC_s_s(distances, X, e, n_obs, n_obs_t, n_vars, n_cores);
      return XeeXh;
    }
  } else {
    if(bartlett) {
      if(flt) {
        // Bartlett, dense, float
        arma::fmat distances(n_obs_t, n_obs_t, arma::fill::zeros);
        dist_mat_f(distances, coords, n_obs_t, dist_cutoff, haversine, n_cores);
        arma::mat XeeXh = XeeXhC_d_f(distances, X, e, n_obs, n_obs_t, n_vars, n_cores);
        return XeeXh;
      } else {
        // Bartlett, dense, double
        arma::mat distances(n_obs_t, n_obs_t, arma::fill::zeros);
        dist_mat_d(distances, coords, n_obs_t, dist_cutoff, haversine, n_cores);
        arma::mat XeeXh = XeeXhC_d_d(distances, X, e, n_obs, n_obs_t, n_vars, n_cores);
        return XeeXh;
      }
    } else {
      // Uniform, dense
      arma::Mat<short int> distances(n_obs_t, n_obs_t, arma::fill::zeros);
      dist_mat_s(distances, coords, n_obs_t, dist_cutoff, haversine, n_cores);
      arma::mat XeeXh = XeeXhC_d_s(distances, X, e, n_obs, n_obs_t, n_vars, n_cores);
      return XeeXh;
    }
  }
}

// 2 Function computing sparse distance matrix with a bartlett kernel and batch insert and calculating spatial sandwich in ols case
// [[Rcpp::export]]
arma::mat ols_d_b(arma::mat &coords, unsigned int n_obs, unsigned int n_obs_t, double dist_cutoff, arma::mat &X, arma::vec &e, unsigned int n_vars, bool haversine,
  unsigned short int batch_ram_opt) {
  std::vector<double> dist_v;
  std::vector<std::vector<unsigned int>> dist_j(n_obs_t);
  double dist {};
  if(haversine) {
    for(unsigned long int i {0}; i < n_obs_t; i++) {
      for(unsigned long int j = i + 1; j < n_obs_t; j++) {
        dist = haversine_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
        if(dist < dist_cutoff) {
          dist = 1.0 - dist / dist_cutoff;
          dist_v.push_back(dist);
          dist_j[i].push_back(j);
        }
      }
    }
  } else {
    for(unsigned long int i {0}; i < n_obs_t; i++) {
      for(unsigned long int j = i + 1; j < n_obs_t; j++) {
        dist = euclidean_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
        if(dist < dist_cutoff) {
          dist = 1.0 - dist / dist_cutoff;
          dist_v.push_back(dist);
          dist_j[i].push_back(j);
        }
      }
    }
  }
  arma::umat index_mat(2, dist_v.size());
  arma::vec values_vec(dist_v);
  if(batch_ram_opt > 1) {
    std::vector<double>().swap(dist_v);
  }
  std::size_t k {0};
  unsigned int vec_length_i {};
  for(unsigned int i {0}; i < n_obs_t; i++) {
    vec_length_i = dist_j[i].size();
    for(std::size_t j {0}; j < vec_length_i; j++) {
      index_mat.at(0, k) = i;
      index_mat.at(1, k) = dist_j[i][j];
      k++;
    }
  }
  if(batch_ram_opt > 1) {
    std::vector<std::vector<unsigned int>>().swap(dist_j);
  }
  arma::sp_mat distances(index_mat, values_vec, n_obs_t, n_obs_t);
  distances.diag().ones();
  distances = arma::symmatu(distances);
  arma::mat XeeXh = XeeXhC_s_d(distances, X, e, n_obs, n_obs_t, n_vars, 1);
  return XeeXh;
}

// 3 Function computing sparse distance matrix with a bartlett kernel and batch insert using floats and calculating spatial sandwich in ols case
// [[Rcpp::export]]
arma::mat ols_f_b(arma::mat &coords, unsigned int n_obs, unsigned int n_obs_t, double dist_cutoff, arma::mat &X, arma::vec &e, unsigned int n_vars, bool haversine,
  unsigned short int batch_ram_opt) {
  std::vector<float> dist_v;
  std::vector<std::vector<unsigned int>> dist_j(n_obs_t);
  double dist {};
  float dist_f {};
  if(haversine) {
    for(unsigned long int i {0}; i < n_obs_t; i++) {
      for(unsigned long int j = i + 1; j < n_obs_t; j++) {
        dist = haversine_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
        if(dist < dist_cutoff) {
          dist_f = (float) (1.0 - dist / dist_cutoff);
          dist_v.push_back(dist_f);
          dist_j[i].push_back(j);
        }
      }
    }
  } else {
    for(unsigned long int i {0}; i < n_obs_t; i++) {
      for(unsigned long int j = i + 1; j < n_obs_t; j++) {
        dist = euclidean_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
        if(dist < dist_cutoff) {
          dist_f = (float) (1.0 - dist / dist_cutoff);
          dist_v.push_back(dist_f);
          dist_j[i].push_back(j);
        }
      }
    }
  }
  arma::umat index_mat(2, dist_v.size());
  arma::fvec values_vec(dist_v);
  if(batch_ram_opt > 1) {
    std::vector<float>().swap(dist_v);
  }
  std::size_t k {0};
  unsigned int vec_length_i {};
  for(unsigned int i {0}; i < n_obs_t; i++) {
    vec_length_i = dist_j[i].size();
    for(std::size_t j {0}; j < vec_length_i; j++) {
      index_mat.at(0, k) = i;
      index_mat.at(1, k) = dist_j[i][j];
      k++;
    }
  }
  if(batch_ram_opt > 1) {
    std::vector<std::vector<unsigned int>>().swap(dist_j);
  }
  arma::sp_fmat distances(index_mat, values_vec, n_obs_t, n_obs_t);
  distances.diag().ones();
  distances = arma::symmatu(distances);
  arma::mat XeeXh = XeeXhC_s_f(distances, X, e, n_obs, n_obs_t, n_vars, 1);
  return XeeXh;
}

// 4 Function parallely computing sparse distance matrix with a bartlett kernel and batch insert and calculating spatial sandwich in ols case
// [[Rcpp::export]]
arma::mat ols_d_b_p(arma::mat &coords, unsigned int n_obs, unsigned int n_obs_t, double dist_cutoff, arma::mat &X, arma::vec &e, unsigned int n_vars, bool haversine,
  unsigned short int batch_ram_opt, unsigned int n_cores) {
  std::vector<std::vector<double>> dist_v(n_obs_t);
  std::vector<std::vector<unsigned int>> dist_j(n_obs_t);
  double dist {};
  unsigned int vec_length_i {};
  arma::Col<unsigned int> vec_intervals(n_obs_t + 1);
  vec_intervals.at(0) = 0;
  std::size_t mat_size {0};
  if(haversine) {
    #pragma omp parallel for private(dist, vec_length_i) schedule(dynamic) reduction(+:mat_size) num_threads(n_cores)
    for(unsigned long int i = 0; i < n_obs_t; i++) {
      for(unsigned long int j = i + 1; j < n_obs_t; j++) {
        dist = haversine_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
        if(dist < dist_cutoff) {
          dist = 1.0 - dist / dist_cutoff;
          dist_v[i].push_back(dist);
          dist_j[i].push_back(j);
        }
      }
      vec_length_i = dist_j[i].size();
      vec_intervals.at(i + 1) = vec_length_i;
      mat_size += vec_length_i;
    }
  } else {
    #pragma omp parallel for private(dist, vec_length_i) schedule(dynamic) reduction(+:mat_size) num_threads(n_cores)
    for(unsigned long int i = 0; i < n_obs_t; i++) {
      for(unsigned long int j = i + 1; j < n_obs_t; j++) {
        dist = euclidean_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
        if(dist < dist_cutoff) {
          dist = 1.0 - dist / dist_cutoff;
          dist_v[i].push_back(dist);
          dist_j[i].push_back(j);
        }
      }
      vec_length_i = dist_j[i].size();
      vec_intervals.at(i + 1) = vec_length_i;
      mat_size += vec_length_i;
    }
  }
  if(mat_size < 4294967296) {
    vec_intervals = arma::cumsum(vec_intervals);
    arma::umat index_mat(2, mat_size);
    arma::vec values_vec(mat_size);
    unsigned int vec_begins_i {};
    unsigned int k {};
    if(batch_ram_opt > 2) {
      #pragma omp parallel for private(k, vec_begins_i, vec_length_i) schedule(dynamic) num_threads(n_cores)
      for(unsigned int i = 0; i < n_obs_t; i++) {
        vec_begins_i = vec_intervals.at(i);
        vec_length_i = vec_intervals.at(i + 1) - vec_begins_i;
        for(unsigned int j = 0; j < vec_length_i; j++) {
          k = vec_begins_i + j;
          index_mat.at(0, k) = i;
          index_mat.at(1, k) = dist_j[i][j];
          values_vec.at(k) = dist_v[i][j];
        }
        std::vector<double>().swap(dist_v[i]);
        std::vector<unsigned int>().swap(dist_j[i]);
      }
      vec_intervals.reset();
    } else {
      #pragma omp parallel for private(k, vec_begins_i, vec_length_i) schedule(dynamic) num_threads(n_cores)
      for(unsigned int i = 0; i < n_obs_t; i++) {
        vec_begins_i = vec_intervals.at(i);
        vec_length_i = vec_intervals.at(i + 1) - vec_begins_i;
        for(unsigned int j = 0; j < vec_length_i; j++) {
          k = vec_begins_i + j;
          index_mat.at(0, k) = i;
          index_mat.at(1, k) = dist_j[i][j];
          values_vec.at(k) = dist_v[i][j];
        }
      }
      if(batch_ram_opt > 1) {
        vec_intervals.reset();
        std::vector<std::vector<double>>().swap(dist_v);
        std::vector<std::vector<unsigned int>>().swap(dist_j);
      }
    }
    arma::sp_mat distances(index_mat, values_vec, n_obs_t, n_obs_t);
    distances.diag().ones();
    distances = arma::symmatu(distances);
    arma::mat XeeXh = XeeXhC_s_d(distances, X, e, n_obs, n_obs_t, n_vars, n_cores);
    return XeeXh;
  } else {
    arma::Col<unsigned long long int> vec_intervals_l = arma::cumsum(arma::conv_to<arma::Col<unsigned long long int>>::from(vec_intervals));
    if(batch_ram_opt > 1) {
      vec_intervals.reset();
    }
    arma::umat index_mat(2, mat_size);
    arma::vec values_vec(mat_size);
    unsigned long long int vec_begins_i {};
    unsigned long long int k {};
    if(batch_ram_opt > 2) {
      #pragma omp parallel for private(k, vec_begins_i, vec_length_i) schedule(dynamic) num_threads(n_cores)
      for(unsigned int i = 0; i < n_obs_t; i++) {
        vec_begins_i = vec_intervals_l.at(i);
        vec_length_i = vec_intervals_l.at(i + 1) - vec_begins_i;
        for(unsigned int j = 0; j < vec_length_i; j++) {
          k = vec_begins_i + j;
          index_mat.at(0, k) = i;
          index_mat.at(1, k) = dist_j[i][j];
          values_vec.at(k) = dist_v[i][j];
        }
        std::vector<double>().swap(dist_v[i]);
        std::vector<unsigned int>().swap(dist_j[i]);
      }
      vec_intervals_l.reset();
    } else {
      #pragma omp parallel for private(k, vec_begins_i, vec_length_i) schedule(dynamic) num_threads(n_cores)
      for(unsigned int i = 0; i < n_obs_t; i++) {
        vec_begins_i = vec_intervals_l.at(i);
        vec_length_i = vec_intervals_l.at(i + 1) - vec_begins_i;
        for(unsigned int j = 0; j < vec_length_i; j++) {
          k = vec_begins_i + j;
          index_mat.at(0, k) = i;
          index_mat.at(1, k) = dist_j[i][j];
          values_vec.at(k) = dist_v[i][j];
        }
      }
      if(batch_ram_opt > 1) {
        vec_intervals_l.reset();
        std::vector<std::vector<double>>().swap(dist_v);
        std::vector<std::vector<unsigned int>>().swap(dist_j);
      }
    }
    arma::sp_mat distances(index_mat, values_vec, n_obs_t, n_obs_t);
    distances.diag().ones();
    distances = arma::symmatu(distances);
    arma::mat XeeXh = XeeXhC_s_d(distances, X, e, n_obs, n_obs_t, n_vars, n_cores);
    return XeeXh;
  }
}

// 5 Function parallely computing sparse distance matrix with a bartlett kernel and batch insert using floats and calculating spatial sandwich in ols case
// [[Rcpp::export]]
arma::mat ols_f_b_p(arma::mat &coords, unsigned int n_obs, unsigned int n_obs_t, double dist_cutoff, arma::mat &X, arma::vec &e, unsigned int n_vars, bool haversine,
  unsigned short int batch_ram_opt, unsigned int n_cores) {
  std::vector<std::vector<float>> dist_v(n_obs_t);
  std::vector<std::vector<unsigned int>> dist_j(n_obs_t);
  double dist {};
  float dist_f {};
  unsigned int vec_length_i {};
  arma::Col<unsigned int> vec_intervals(n_obs_t + 1);
  vec_intervals.at(0) = 0;
  std::size_t mat_size {0};
  if(haversine) {
    #pragma omp parallel for private(dist, dist_f, vec_length_i) schedule(dynamic) reduction(+:mat_size) num_threads(n_cores)
    for(unsigned long int i = 0; i < n_obs_t; i++) {
      for(unsigned long int j = i + 1; j < n_obs_t; j++) {
        dist = haversine_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
        if(dist < dist_cutoff) {
          dist_f = (float) (1.0 - dist / dist_cutoff);
          dist_v[i].push_back(dist_f);
          dist_j[i].push_back(j);
        }
      }
      vec_length_i = dist_j[i].size();
      vec_intervals.at(i + 1) = vec_length_i;
      mat_size += vec_length_i;
    }
  } else {
    #pragma omp parallel for private(dist, dist_f, vec_length_i) schedule(dynamic) reduction(+:mat_size) num_threads(n_cores)
    for(unsigned long int i = 0; i < n_obs_t; i++) {
      for(unsigned long int j = i + 1; j < n_obs_t; j++) {
        dist = euclidean_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
        if(dist < dist_cutoff) {
          dist_f = (float) (1.0 - dist / dist_cutoff);
          dist_v[i].push_back(dist_f);
          dist_j[i].push_back(j);
        }
      }
      vec_length_i = dist_j[i].size();
      vec_intervals.at(i + 1) = vec_length_i;
      mat_size += vec_length_i;
    }
  }
  if(mat_size < 4294967296) {
    vec_intervals = arma::cumsum(vec_intervals);
    arma::umat index_mat(2, mat_size);
    arma::fvec values_vec(mat_size);
    unsigned int vec_begins_i {};
    unsigned int k {};
    if(batch_ram_opt > 2) {
      #pragma omp parallel for private(k, vec_begins_i, vec_length_i) schedule(dynamic) num_threads(n_cores)
      for(unsigned int i = 0; i < n_obs_t; i++) {
        vec_begins_i = vec_intervals.at(i);
        vec_length_i = vec_intervals.at(i + 1) - vec_begins_i;
        for(unsigned int j = 0; j < vec_length_i; j++) {
          k = vec_begins_i + j;
          index_mat.at(0, k) = i;
          index_mat.at(1, k) = dist_j[i][j];
          values_vec.at(k) = dist_v[i][j];
        }
        std::vector<float>().swap(dist_v[i]);
        std::vector<unsigned int>().swap(dist_j[i]);
      }
      vec_intervals.reset();
    } else {
      #pragma omp parallel for private(k, vec_begins_i, vec_length_i) schedule(dynamic) num_threads(n_cores)
      for(unsigned int i = 0; i < n_obs_t; i++) {
        vec_begins_i = vec_intervals.at(i);
        vec_length_i = vec_intervals.at(i + 1) - vec_begins_i;
        for(unsigned int j = 0; j < vec_length_i; j++) {
          k = vec_begins_i + j;
          index_mat.at(0, k) = i;
          index_mat.at(1, k) = dist_j[i][j];
          values_vec.at(k) = dist_v[i][j];
        }
      }
      if(batch_ram_opt > 1) {
        vec_intervals.reset();
        std::vector<std::vector<float>>().swap(dist_v);
        std::vector<std::vector<unsigned int>>().swap(dist_j);
      }
    }
    arma::sp_fmat distances(index_mat, values_vec, n_obs_t, n_obs_t);
    distances.diag().ones();
    distances = arma::symmatu(distances);
    arma::mat XeeXh = XeeXhC_s_f(distances, X, e, n_obs, n_obs_t, n_vars, n_cores);
    return XeeXh;
  } else {
    arma::Col<unsigned long long int> vec_intervals_l = arma::cumsum(arma::conv_to<arma::Col<unsigned long long int>>::from(vec_intervals));
    if(batch_ram_opt > 1) {
      vec_intervals.reset();
    }
    arma::umat index_mat(2, mat_size);
    arma::fvec values_vec(mat_size);
    unsigned long long int vec_begins_i {};
    unsigned long long int k {};
    if(batch_ram_opt > 2) {
      #pragma omp parallel for private(k, vec_begins_i, vec_length_i) schedule(dynamic) num_threads(n_cores)
      for(unsigned int i = 0; i < n_obs_t; i++) {
        vec_begins_i = vec_intervals_l.at(i);
        vec_length_i = vec_intervals_l.at(i + 1) - vec_begins_i;
        for(unsigned int j = 0; j < vec_length_i; j++) {
          k = vec_begins_i + j;
          index_mat.at(0, k) = i;
          index_mat.at(1, k) = dist_j[i][j];
          values_vec.at(k) = dist_v[i][j];
        }
        std::vector<float>().swap(dist_v[i]);
        std::vector<unsigned int>().swap(dist_j[i]);
      }
      vec_intervals_l.reset();
    } else {
      #pragma omp parallel for private(k, vec_begins_i, vec_length_i) schedule(dynamic) num_threads(n_cores)
      for(unsigned int i = 0; i < n_obs_t; i++) {
        vec_begins_i = vec_intervals_l.at(i);
        vec_length_i = vec_intervals_l.at(i + 1) - vec_begins_i;
        for(unsigned int j = 0; j < vec_length_i; j++) {
          k = vec_begins_i + j;
          index_mat.at(0, k) = i;
          index_mat.at(1, k) = dist_j[i][j];
          values_vec.at(k) = dist_v[i][j];
        }
      }
      if(batch_ram_opt > 1) {
        vec_intervals_l.reset();
        std::vector<std::vector<float>>().swap(dist_v);
        std::vector<std::vector<unsigned int>>().swap(dist_j);
      }
    }
    arma::sp_fmat distances(index_mat, values_vec, n_obs_t, n_obs_t);
    distances.diag().ones();
    distances = arma::symmatu(distances);
    arma::mat XeeXh = XeeXhC_s_f(distances, X, e, n_obs, n_obs_t, n_vars, n_cores);
    return XeeXh;
  }
}

// 6 Function computing sparse distance matrix with a uniform kernel and batch insert and calculating spatial sandwich in ols case
// [[Rcpp::export]]
arma::mat ols_s_b(arma::mat &coords, unsigned int n_obs, unsigned int n_obs_t, double dist_cutoff, arma::mat &X, arma::vec &e, unsigned int n_vars, bool haversine,
  unsigned short int batch_ram_opt) {
  std::vector<short int> dist_v;
  std::vector<std::vector<unsigned int>> dist_j(n_obs_t);
  double dist {};
  if(haversine) {
    for(unsigned long int i {0}; i < n_obs_t; i++) {
      for(unsigned long int j = i + 1; j < n_obs_t; j++) {
        dist = haversine_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
        if(dist < dist_cutoff) {
          dist_v.push_back(1);
          dist_j[i].push_back(j);
        }
      }
    }
  } else {
    for(unsigned long int i {0}; i < n_obs_t; i++) {
      for(unsigned long int j = i + 1; j < n_obs_t; j++) {
        dist = euclidean_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
        if(dist < dist_cutoff) {
          dist_v.push_back(1);
          dist_j[i].push_back(j);
        }
      }
    }
  }
  arma::umat index_mat(2, dist_v.size());
  arma::Col<short int> values_vec(dist_v);
  if(batch_ram_opt > 1) {
    std::vector<short int>().swap(dist_v);
  }
  std::size_t k {0};
  unsigned int vec_length_i {};
  for(unsigned int i {0}; i < n_obs_t; i++) {
    vec_length_i = dist_j[i].size();
    for(std::size_t j {0}; j < vec_length_i; j++) {
      index_mat.at(0, k) = i;
      index_mat.at(1, k) = dist_j[i][j];
      k++;
    }
  }
  if(batch_ram_opt > 1) {
    std::vector<std::vector<unsigned int>>().swap(dist_j);
  }
  arma::SpMat<short int> distances(index_mat, values_vec, n_obs_t, n_obs_t);
  distances.diag().ones();
  distances = arma::symmatu(distances);
  arma::mat XeeXh = XeeXhC_s_s(distances, X, e, n_obs, n_obs_t, n_vars, 1);
  return XeeXh;
}

// 7 Function parallely computing sparse distance matrix with a uniform kernel and batch insert and calculating spatial sandwich in ols case
// [[Rcpp::export]]
arma::mat ols_s_b_p(arma::mat &coords, unsigned int n_obs, unsigned int n_obs_t, double dist_cutoff, arma::mat &X, arma::vec &e, unsigned int n_vars, bool haversine,
  unsigned short int batch_ram_opt, unsigned int n_cores) {
  std::vector<std::vector<short int>> dist_v(n_obs_t);
  std::vector<std::vector<unsigned int>> dist_j(n_obs_t);
  double dist {};
  unsigned int vec_length_i {};
  arma::Col<unsigned int> vec_intervals(n_obs_t + 1);
  vec_intervals.at(0) = 0;
  std::size_t mat_size {0};
  if(haversine) {
    #pragma omp parallel for private(dist, vec_length_i) schedule(dynamic) reduction(+:mat_size) num_threads(n_cores)
    for(unsigned long int i = 0; i < n_obs_t; i++) {
      for(unsigned long int j = i + 1; j < n_obs_t; j++) {
        dist = haversine_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
        if(dist < dist_cutoff) {
          dist_v[i].push_back(1);
          dist_j[i].push_back(j);
        }
      }
      vec_length_i = dist_j[i].size();
      vec_intervals.at(i + 1) = vec_length_i;
      mat_size += vec_length_i;
    }
  } else {
    #pragma omp parallel for private(dist, vec_length_i) schedule(dynamic) reduction(+:mat_size) num_threads(n_cores)
    for(unsigned long int i = 0; i < n_obs_t; i++) {
      for(unsigned long int j = i + 1; j < n_obs_t; j++) {
        dist = euclidean_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
        if(dist < dist_cutoff) {
          dist_v[i].push_back(1);
          dist_j[i].push_back(j);
        }
      }
      vec_length_i = dist_j[i].size();
      vec_intervals.at(i + 1) = vec_length_i;
      mat_size += vec_length_i;
    }
  }
  if(mat_size < 4294967296) {
    vec_intervals = arma::cumsum(vec_intervals);
    arma::umat index_mat(2, mat_size);
    arma::Col<short int> values_vec(mat_size);
    unsigned int vec_begins_i {};
    unsigned int k {};
    if(batch_ram_opt > 2) {
      #pragma omp parallel for private(k, vec_begins_i, vec_length_i) schedule(dynamic) num_threads(n_cores)
      for(unsigned int i = 0; i < n_obs_t; i++) {
        vec_begins_i = vec_intervals.at(i);
        vec_length_i = vec_intervals.at(i + 1) - vec_begins_i;
        for(unsigned int j = 0; j < vec_length_i; j++) {
          k = vec_begins_i + j;
          index_mat.at(0, k) = i;
          index_mat.at(1, k) = dist_j[i][j];
          values_vec.at(k) = dist_v[i][j];
        }
        std::vector<short int>().swap(dist_v[i]);
        std::vector<unsigned int>().swap(dist_j[i]);
      }
      vec_intervals.reset();
    } else {
      #pragma omp parallel for private(k, vec_begins_i, vec_length_i) schedule(dynamic) num_threads(n_cores)
      for(unsigned int i = 0; i < n_obs_t; i++) {
        vec_begins_i = vec_intervals.at(i);
        vec_length_i = vec_intervals.at(i + 1) - vec_begins_i;
        for(unsigned int j = 0; j < vec_length_i; j++) {
          k = vec_begins_i + j;
          index_mat.at(0, k) = i;
          index_mat.at(1, k) = dist_j[i][j];
          values_vec.at(k) = dist_v[i][j];
        }
      }
      if(batch_ram_opt > 1) {
        vec_intervals.reset();
        std::vector<std::vector<short int>>().swap(dist_v);
        std::vector<std::vector<unsigned int>>().swap(dist_j);
      }
    }
    arma::SpMat<short int> distances(index_mat, values_vec, n_obs_t, n_obs_t);
    distances.diag().ones();
    distances = arma::symmatu(distances);
    arma::mat XeeXh = XeeXhC_s_s(distances, X, e, n_obs, n_obs_t, n_vars, n_cores);
    return XeeXh;
  } else {
    arma::Col<unsigned long long int> vec_intervals_l = arma::cumsum(arma::conv_to<arma::Col<unsigned long long int>>::from(vec_intervals));
    if(batch_ram_opt > 1) {
      vec_intervals.reset();
    }
    arma::umat index_mat(2, mat_size);
    arma::Col<short int> values_vec(mat_size);
    unsigned long long int vec_begins_i {};
    unsigned long long int k {};
    if(batch_ram_opt > 2) {
      #pragma omp parallel for private(k, vec_begins_i, vec_length_i) schedule(dynamic) num_threads(n_cores)
      for(unsigned int i = 0; i < n_obs_t; i++) {
        vec_begins_i = vec_intervals_l.at(i);
        vec_length_i = vec_intervals_l.at(i + 1) - vec_begins_i;
        for(unsigned int j = 0; j < vec_length_i; j++) {
          k = vec_begins_i + j;
          index_mat.at(0, k) = i;
          index_mat.at(1, k) = dist_j[i][j];
          values_vec.at(k) = dist_v[i][j];
        }
        std::vector<short int>().swap(dist_v[i]);
        std::vector<unsigned int>().swap(dist_j[i]);
      }
      vec_intervals_l.reset();
    } else {
      #pragma omp parallel for private(k, vec_begins_i, vec_length_i) schedule(dynamic) num_threads(n_cores)
      for(unsigned int i = 0; i < n_obs_t; i++) {
        vec_begins_i = vec_intervals_l.at(i);
        vec_length_i = vec_intervals_l.at(i + 1) - vec_begins_i;
        for(unsigned int j = 0; j < vec_length_i; j++) {
          k = vec_begins_i + j;
          index_mat.at(0, k) = i;
          index_mat.at(1, k) = dist_j[i][j];
          values_vec.at(k) = dist_v[i][j];
        }
      }
      if(batch_ram_opt > 1) {
        vec_intervals_l.reset();
        std::vector<std::vector<short int>>().swap(dist_v);
        std::vector<std::vector<unsigned int>>().swap(dist_j);
      }
    }
    arma::SpMat<short int> distances(index_mat, values_vec, n_obs_t, n_obs_t);
    distances.diag().ones();
    distances = arma::symmatu(distances);
    arma::mat XeeXh = XeeXhC_s_s(distances, X, e, n_obs, n_obs_t, n_vars, n_cores);
    return XeeXh;
  }
}

// 8 Function computing row-wise distances and calculating spatial sandwich in ols case
// [[Rcpp::export]]
arma::mat ols_r(arma::mat &coords, unsigned int n_obs, double dist_cutoff, arma::mat &X, arma::vec &e, unsigned int n_vars, bool haversine, bool bartlett,
  bool flt, unsigned int n_cores) {
  // Set empty output matrix
  arma::mat XeeXh(n_vars, n_vars, arma::fill::zeros);
  // Generate a [N variables x 1] matrix with all values set to one
  arma::mat k_mat(n_vars, 1, arma::fill::ones);
  double dist {};
  if(bartlett) {
    if(flt) {
      arma::fmat d_row(1, n_obs, arma::fill::ones);
      if(haversine) {
        if(n_cores > 1) {
          #pragma omp parallel for private(dist) num_threads(n_cores)
          for(unsigned int i = 0; i < n_cores; i++) {
            arma::mat XeeXh_i(n_vars, n_vars, arma::fill::zeros);
            arma::mat e_mat(1, n_obs, arma::fill::zeros);
            for(unsigned int j = i; j < n_obs; j += n_cores) {
              arma::fmat distances(1, n_obs, arma::fill::zeros);
              for(unsigned int k = 0; k < n_obs; k++) {
                if(k == j) {
                  distances.at(0, k) = 1.0;
                } else {
                  dist = haversine_dist(coords(j, 1), coords(k, 1), coords(j, 0), coords(k, 0));
                  if(dist < dist_cutoff) {
                    distances.at(0, k) = (float) (1.0 - dist / dist_cutoff);
                  }
                }
              }
              e_mat.fill(e[j]);
              XeeXh_i += ((k_mat % X.row(j).t()) * e_mat % (k_mat * ((d_row % distances) % e.t()))) * X;
            }
            #pragma omp critical
            XeeXh += XeeXh_i;
          }
        } else {
          arma::mat e_mat(1, n_obs, arma::fill::zeros);
          for(unsigned int i {0}; i < n_obs; i++) {
            arma::fmat distances(1, n_obs, arma::fill::zeros);
            for(unsigned int j {0}; j < n_obs; j++) {
              if(j == i) {
                distances.at(0, j) = 1.0;
              } else {
                dist = haversine_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
                if(dist < dist_cutoff) {
                  distances.at(0, j) = (float) (1.0 - dist / dist_cutoff);
                }
              }
            }
            e_mat.fill(e[i]);
            XeeXh += ((k_mat % X.row(i).t()) * e_mat % (k_mat * ((d_row % distances) % e.t()))) * X;
          }
        }
      } else {
        if(n_cores > 1) {
          #pragma omp parallel for private(dist) num_threads(n_cores)
          for(unsigned int i = 0; i < n_cores; i++) {
            arma::mat XeeXh_i(n_vars, n_vars, arma::fill::zeros);
            arma::mat e_mat(1, n_obs, arma::fill::zeros);
            for(unsigned int j = i; j < n_obs; j += n_cores) {
              arma::fmat distances(1, n_obs, arma::fill::zeros);
              for(unsigned int k = 0; k < n_obs; k++) {
                if(k == j) {
                  distances.at(0, k) = 1.0;
                } else {
                  dist = euclidean_dist(coords(j, 1), coords(k, 1), coords(j, 0), coords(k, 0));
                  if(dist < dist_cutoff) {
                    distances.at(0, k) = (float) (1.0 - dist / dist_cutoff);
                  }
                }
              }
              e_mat.fill(e[j]);
              XeeXh_i += ((k_mat % X.row(j).t()) * e_mat % (k_mat * ((d_row % distances) % e.t()))) * X;
            }
            #pragma omp critical
            XeeXh += XeeXh_i;
          }
        } else {
          arma::mat e_mat(1, n_obs, arma::fill::zeros);
          for(unsigned int i {0}; i < n_obs; i++) {
            arma::fmat distances(1, n_obs, arma::fill::zeros);
            for(unsigned int j {0}; j < n_obs; j++) {
              if(j == i) {
                distances.at(0, j) = 1.0;
              } else {
                dist = euclidean_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
                if(dist < dist_cutoff) {
                  distances.at(0, j) = (float) (1.0 - dist / dist_cutoff);
                }
              }
            }
            e_mat.fill(e[i]);
            XeeXh += ((k_mat % X.row(i).t()) * e_mat % (k_mat * ((d_row % distances) % e.t()))) * X;
          }
        }
      }
    } else {
      arma::mat d_row(1, n_obs, arma::fill::ones);
      if(haversine) {
        if(n_cores > 1) {
          #pragma omp parallel for private(dist) num_threads(n_cores)
          for(unsigned int i = 0; i < n_cores; i++) {
            arma::mat XeeXh_i(n_vars, n_vars, arma::fill::zeros);
            arma::mat e_mat(1, n_obs, arma::fill::zeros);
            for(unsigned int j = i; j < n_obs; j += n_cores) {
              arma::mat distances(1, n_obs, arma::fill::zeros);
              for(unsigned int k = 0; k < n_obs; k++) {
                if(k == j) {
                  distances.at(0, k) = 1.0;
                } else {
                  dist = haversine_dist(coords(j, 1), coords(k, 1), coords(j, 0), coords(k, 0));
                  if(dist < dist_cutoff) {
                    distances.at(0, k) = 1.0 - dist / dist_cutoff;
                  }
                }
              }
              e_mat.fill(e[j]);
              XeeXh_i += ((k_mat % X.row(j).t()) * e_mat % (k_mat * ((d_row % distances) % e.t()))) * X;
            }
            #pragma omp critical
            XeeXh += XeeXh_i;
          }
        } else {
          arma::mat e_mat(1, n_obs, arma::fill::zeros);
          for(unsigned int i {0}; i < n_obs; i++) {
            arma::mat distances(1, n_obs, arma::fill::zeros);
            for(unsigned int j {0}; j < n_obs; j++) {
              if(j == i) {
                distances.at(0, j) = 1.0;
              } else {
                dist = haversine_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
                if(dist < dist_cutoff) {
                  distances.at(0, j) = 1.0 - dist / dist_cutoff;
                }
              }
            }
            e_mat.fill(e[i]);
            XeeXh += ((k_mat % X.row(i).t()) * e_mat % (k_mat * ((d_row % distances) % e.t()))) * X;
          }
        }
      } else {
        if(n_cores > 1) {
          #pragma omp parallel for private(dist) num_threads(n_cores)
          for(unsigned int i = 0; i < n_cores; i++) {
            arma::mat XeeXh_i(n_vars, n_vars, arma::fill::zeros);
            arma::mat e_mat(1, n_obs, arma::fill::zeros);
            for(unsigned int j = i; j < n_obs; j += n_cores) {
              arma::mat distances(1, n_obs, arma::fill::zeros);
              for(unsigned int k = 0; k < n_obs; k++) {
                if(k == j) {
                  distances.at(0, k) = 1.0;
                } else {
                  dist = euclidean_dist(coords(j, 1), coords(k, 1), coords(j, 0), coords(k, 0));
                  if(dist < dist_cutoff) {
                    distances.at(0, k) = 1.0 - dist / dist_cutoff;
                  }
                }
              }
              e_mat.fill(e[j]);
              XeeXh_i += ((k_mat % X.row(j).t()) * e_mat % (k_mat * ((d_row % distances) % e.t()))) * X;
            }
            #pragma omp critical
            XeeXh += XeeXh_i;
          }
        } else {
          arma::mat e_mat(1, n_obs, arma::fill::zeros);
          for(unsigned int i {0}; i < n_obs; i++) {
            arma::mat distances(1, n_obs, arma::fill::zeros);
            for(unsigned int j {0}; j < n_obs; j++) {
              if(j == i) {
                distances.at(0, j) = 1.0;
              } else {
                dist = euclidean_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
                if(dist < dist_cutoff) {
                  distances.at(0, j) = 1.0 - dist / dist_cutoff;
                }
              }
            }
            e_mat.fill(e[i]);
            XeeXh += ((k_mat % X.row(i).t()) * e_mat % (k_mat * ((d_row % distances) % e.t()))) * X;
          }
        }
      }
    }
  } else {
    arma::Mat<short int> d_row(1, n_obs, arma::fill::ones);
    if(haversine) {
      if(n_cores > 1) {
        #pragma omp parallel for private(dist) num_threads(n_cores)
        for(unsigned int i = 0; i < n_cores; i++) {
          arma::mat XeeXh_i(n_vars, n_vars, arma::fill::zeros);
          arma::mat e_mat(1, n_obs, arma::fill::zeros);
          for(unsigned int j = i; j < n_obs; j += n_cores) {
            arma::Mat<short int> distances(1, n_obs, arma::fill::zeros);
            for(unsigned int k = 0; k < n_obs; k++) {
              if(k == j) {
                distances.at(0, k) = 1;
              } else {
                dist = haversine_dist(coords(j, 1), coords(k, 1), coords(j, 0), coords(k, 0));
                if(dist < dist_cutoff) {
                  distances.at(0, k) = 1;
                }
              }
            }
            e_mat.fill(e[j]);
            XeeXh_i += ((k_mat % X.row(j).t()) * e_mat % (k_mat * ((d_row % distances) % e.t()))) * X;
          }
          #pragma omp critical
          XeeXh += XeeXh_i;
        }
      } else {
        arma::mat e_mat(1, n_obs, arma::fill::zeros);
        for(unsigned int i {0}; i < n_obs; i++) {
          arma::Mat<short int> distances(1, n_obs, arma::fill::zeros);
          for(unsigned int j {0}; j < n_obs; j++) {
            if(j == i) {
              distances.at(0, j) = 1;
            } else {
              dist = haversine_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
              if(dist < dist_cutoff) {
                distances.at(0, j) = 1;
              }
            }
          }
          e_mat.fill(e[i]);
          XeeXh += ((k_mat % X.row(i).t()) * e_mat % (k_mat * ((d_row % distances) % e.t()))) * X;
        }
      }
    } else {
      if(n_cores > 1) {
        #pragma omp parallel for private(dist) num_threads(n_cores)
        for(unsigned int i = 0; i < n_cores; i++) {
          arma::mat XeeXh_i(n_vars, n_vars, arma::fill::zeros);
          arma::mat e_mat(1, n_obs, arma::fill::zeros);
          for(unsigned int j = i; j < n_obs; j += n_cores) {
            arma::Mat<short int> distances(1, n_obs, arma::fill::zeros);
            for(unsigned int k = 0; k < n_obs; k++) {
              if(k == j) {
              distances.at(0, k) = 1;
            } else {
                dist = euclidean_dist(coords(j, 1), coords(k, 1), coords(j, 0), coords(k, 0));
                if(dist < dist_cutoff) {
                  distances.at(0, k) = 1;
                }
              }
            }
            e_mat.fill(e[j]);
            XeeXh_i += ((k_mat % X.row(j).t()) * e_mat % (k_mat * ((d_row % distances) % e.t()))) * X;
          }
          #pragma omp critical
          XeeXh += XeeXh_i;
        }
      } else {
        arma::mat e_mat(1, n_obs, arma::fill::zeros);
        for(unsigned int i {0}; i < n_obs; i++) {
          arma::Mat<short int> distances(1, n_obs, arma::fill::zeros);
          for(unsigned int j {0}; j < n_obs; j++) {
            if(j == i) {
              distances.at(0, j) = 1;
            } else {
              dist = euclidean_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
              if(dist < dist_cutoff) {
                distances.at(0, j) = 1;
              }
            }
          }
          e_mat.fill(e[i]);
          XeeXh += ((k_mat % X.row(i).t()) * e_mat % (k_mat * ((d_row % distances) % e.t()))) * X;
        }
      }
    }
  }
  return XeeXh;
}
