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
#include "distance_functions.h"
#include "distance_matrices.h"

// Function overview
// 1 dist_mat_d Dense distance matrix with bartlett kernel in doubles
// 2 dist_mat_f Dense distance matrix with bartlett kernel in floats
// 3 dist_mat_s Dense distance matrix with uniform kernel
// 4 dist_spmat_d Sparse distance matrix with bartlett kernel in doubles
// 5 dist_spmat_f Sparse distance matrix with bartlett kernel in floats
// 6 dist_spmat_s Sparse distance matrix with uniform kernel

// 1 Dense distance matrix with bartlett kernel in doubles
void dist_mat_d(arma::mat &distances, arma::mat &coords, unsigned int n_obs_t, double dist_cutoff, bool haversine, unsigned int n_cores) {
  double dist {};
  if(haversine) {
    #pragma omp parallel for private(dist) schedule(dynamic) num_threads(n_cores) if(n_cores > 1)
    for(unsigned int i = 0; i < n_obs_t; i++) {
      for(unsigned int j = i + 1; j < n_obs_t; j++) {
        dist = haversine_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
        if(dist < dist_cutoff) {
          dist = 1.0 - dist / dist_cutoff;
          distances.at(i, j) = dist;
          distances.at(j, i) = dist;
        }
      }
    }
  } else {
    #pragma omp parallel for private(dist) schedule(dynamic) num_threads(n_cores) if(n_cores > 1)
    for(unsigned int i = 0; i < n_obs_t; i++) {
      for(unsigned int j = i + 1; j < n_obs_t; j++) {
        dist = euclidean_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
        if(dist < dist_cutoff) {
          dist = 1.0 - dist / dist_cutoff;
          distances.at(i, j) = dist;
          distances.at(j, i) = dist;
        }
      }
    }
  }
  distances.diag().ones();
}

// 2 Dense distance matrix with bartlett kernel in floats
void dist_mat_f(arma::fmat &distances, arma::mat &coords, unsigned int n_obs_t, double dist_cutoff, bool haversine, unsigned int n_cores) {
  double dist {};
  float dist_f{};
  if(haversine) {
    #pragma omp parallel for private(dist, dist_f) schedule(dynamic) num_threads(n_cores) if(n_cores > 1)
    for(unsigned int i = 0; i < n_obs_t; i++) {
      for(unsigned int j = i + 1; j < n_obs_t; j++) {
        dist = haversine_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
        if(dist < dist_cutoff) {
          dist_f = (float) (1.0 - dist / dist_cutoff);
          distances.at(i, j) = dist_f;
          distances.at(j, i) = dist_f;
        }
      }
    }
  } else {
    #pragma omp parallel for private(dist, dist_f) schedule(dynamic) num_threads(n_cores) if(n_cores > 1)
    for(unsigned int i = 0; i < n_obs_t; i++) {
      for(unsigned int j = i + 1; j < n_obs_t; j++) {
        dist = euclidean_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
        if(dist < dist_cutoff) {
          dist_f = (float) (1.0 - dist / dist_cutoff);
          distances.at(i, j) = dist_f;
          distances.at(j, i) = dist_f;
        }
      }
    }
  }
  distances.diag().ones();
}

// 3 Dense distance matrix with uniform kernel
void dist_mat_s(arma::Mat<short int> &distances, arma::mat &coords, unsigned int n_obs_t, double dist_cutoff, bool haversine, unsigned int n_cores) {
  double dist {};
  if(haversine) {
    #pragma omp parallel for private(dist) schedule(dynamic) num_threads(n_cores) if(n_cores > 1)
    for(unsigned int i = 0; i < n_obs_t; i++) {
      for(unsigned int j = i + 1; j < n_obs_t; j++) {
        dist = haversine_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
        if(dist < dist_cutoff) {
          distances.at(i, j) = 1;
          distances.at(j, i) = 1;
        }
      }
    }
  } else {
    #pragma omp parallel for private(dist) schedule(dynamic) num_threads(n_cores) if(n_cores > 1)
    for(unsigned int i = 0; i < n_obs_t; i++) {
      for(unsigned int j = i + 1; j < n_obs_t; j++) {
        dist = euclidean_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
        if(dist < dist_cutoff) {
          distances.at(i, j) = 1;
          distances.at(j, i) = 1;
        }
      }
    }
  }
  distances.diag().ones();
}

// 4 Sparse distance matrix with bartlett kernel in doubles
void dist_spmat_d(arma::sp_mat &distances, arma::mat &coords, unsigned int n_obs_t, double dist_cutoff, bool haversine, unsigned int n_cores) {
  double dist {};
  if(haversine) {
    #pragma omp parallel for private(dist) schedule(dynamic) num_threads(n_cores) if(n_cores > 1)
    for(unsigned int i = 0; i < n_obs_t; i++) {
      for(unsigned int j = i + 1; j < n_obs_t; j++) {
        dist = haversine_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
        if(dist < dist_cutoff) {
          dist = 1.0 - dist / dist_cutoff;
          distances.at(i, j) = dist;
        }
      }
    }
  } else {
    #pragma omp parallel for private(dist) schedule(dynamic) num_threads(n_cores) if(n_cores > 1)
    for(unsigned int i = 0; i < n_obs_t; i++) {
      for(unsigned int j = i + 1; j < n_obs_t; j++) {
        dist = euclidean_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
        if(dist < dist_cutoff) {
          dist = 1.0 - dist / dist_cutoff;
          distances.at(i, j) = dist;
        }
      }
    }
  }
  distances = arma::symmatu(distances);
  distances.diag().ones();
}

// 5 Sparse distance matrix with bartlett kernel in floats
void dist_spmat_f(arma::sp_fmat &distances, arma::mat &coords, unsigned int n_obs_t, double dist_cutoff, bool haversine, unsigned int n_cores) {
  double dist {};
  float dist_f{};
  if(haversine) {
    #pragma omp parallel for private(dist, dist_f) schedule(dynamic) num_threads(n_cores) if(n_cores > 1)
    for(unsigned int i = 0; i < n_obs_t; i++) {
      for(unsigned int j = i + 1; j < n_obs_t; j++) {
        dist = haversine_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
        if(dist < dist_cutoff) {
          dist_f = (float) (1.0 - dist / dist_cutoff);
          distances.at(i, j) = dist_f;
        }
      }
    }
  } else {
    #pragma omp parallel for private(dist, dist_f) schedule(dynamic) num_threads(n_cores) if(n_cores > 1)
    for(unsigned int i = 0; i < n_obs_t; i++) {
      for(unsigned int j = i + 1; j < n_obs_t; j++) {
        dist = euclidean_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
        if(dist < dist_cutoff) {
          dist_f = (float) (1.0 - dist / dist_cutoff);
          distances.at(i, j) = dist_f;
        }
      }
    }
  }
  distances = arma::symmatu(distances);
  distances.diag().ones();
}

// 6 Sparse distance matrix with uniform kernel
void dist_spmat_s(arma::SpMat<short int> &distances, arma::mat &coords, unsigned int n_obs_t, double dist_cutoff, bool haversine, unsigned int n_cores) {
  double dist {};
  if(haversine) {
    #pragma omp parallel for private(dist) schedule(dynamic) num_threads(n_cores) if(n_cores > 1)
    for(unsigned int i = 0; i < n_obs_t; i++) {
      for(unsigned int j = i + 1; j < n_obs_t; j++) {
        dist = haversine_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
        if(dist < dist_cutoff) {
          distances.at(i, j) = 1;
        }
      }
    }
  } else {
    #pragma omp parallel for private(dist) schedule(dynamic) num_threads(n_cores) if(n_cores > 1)
    for(unsigned int i = 0; i < n_obs_t; i++) {
      for(unsigned int j = i + 1; j < n_obs_t; j++) {
        dist = euclidean_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
        if(dist < dist_cutoff) {
          distances.at(i, j) = 1;
        }
      }
    }
  }
  distances = arma::symmatu(distances);
  distances.diag().ones();
}
