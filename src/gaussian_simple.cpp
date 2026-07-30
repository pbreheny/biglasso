#include "utilities.h"

// Coordinate descent for gaussian models; no adaptation or SSR
// NOTE: these functions do not standardize X or center y

// apply the requested penalty to a single coordinate
static double update_beta(const char *penalty, double z, double l1, double l2,
                          double gamma, double v) {
  if (strcmp(penalty, "MCP") == 0) return MCP(z, l1, l2, gamma, v);
  if (strcmp(penalty, "SCAD") == 0) return SCAD(z, l1, l2, gamma, v);
  return lasso(z, l1, l2, v);
}

// shared coordinate-descent path solver behind cdfit_gaussian_simple() and
// cdfit_gaussian_simple_path(); the former just calls this with a length-1
// lambda vector and a dfmax that can never be triggered (p features max)
static arma::sp_mat cdfit_gaussian_simple_core(XPtr<BigMatrix> xMat, double *r, double *init,
                                               double *xtx, const char *penalty,
                                               NumericVector &lambda, double alpha, double gamma,
                                               double thresh, int dfmax, int max_iter, double *m,
                                               int n, int p, NumericVector &loss,
                                               IntegerVector &iter) {
  int L = lambda.size();
  arma::sp_mat beta(p, L);
  double *a = R_Calloc(p, double); // will hold beta from previous iteration
  int *ever_active = R_Calloc(p, int); // ever-active set
  NumericVector z(p);
  double l1, l2, shift, cp, max_update, update;
  int j, l;

  for (j = 0; j < p; j++) {
    a[j] = init[j];
    ever_active[j] = 1 * (a[j] != 0);
  }

  for (l = 0; l < L; l++) {
    while (iter[l] < max_iter) {
      R_CheckUserInterrupt();
      while (iter[l] < max_iter) {
        iter[l]++;
        max_update = 0.0;

        // solve over active set
        for (j = 0; j < p; j++) {
          if (ever_active[j]) {
            cp = crossprod_bm_no_std(xMat, r, n, j);
            z[j] = cp / n + xtx[j] * a[j];

            l1 = lambda[l] * m[j] * alpha;
            l2 = lambda[l] * m[j] * (1 - alpha);
            beta(j, l) = update_beta(penalty, z[j], l1, l2, gamma, xtx[j]);

            // update residuals
            shift = beta(j, l) - a[j];
            if (shift != 0) {
              update_resid_no_std(xMat, r, shift, n, j);
              update = fabs(shift) * sqrt(xtx[j]);
              if (update > max_update) max_update = update;
            }
          }
        }
        // make current beta the old value
        for (j = 0; j < p; j++) a[j] = beta(j, l);

        // check for convergence
        if (max_update < thresh) break;
      }

      // scan for violations
      int violations = 0;
      for (j = 0; j < p; j++) {
        if (!ever_active[j]) {
          z[j] = crossprod_bm_no_std(xMat, r, n, j) / n;

          l1 = lambda[l] * m[j] * alpha;
          l2 = lambda[l] * m[j] * (1 - alpha);
          beta(j, l) = update_beta(penalty, z[j], l1, l2, gamma, xtx[j]);

          // if something enters, update active set and residuals
          if (beta(j, l) != 0) {
            ever_active[j] = 1;
            update_resid_no_std(xMat, r, beta(j, l), n, j);
            a[j] = beta(j, l);
            violations++;
          }
        }
      }
      if (violations == 0) break;
    }

    loss[l] = gLoss(r, n);

    // check dfmax
    int nv = 0;
    for (j = 0; j < p; j++) {
      if (a[j] != 0) nv++;
    }
    if (nv > dfmax) {
      for (int ll = l; ll < L; ll++) iter[ll] = NA_INTEGER;
      break;
    }
  }

  R_Free(a);
  R_Free(ever_active);
  return beta;
}

static void set_omp_cores(int ncore) {
#ifdef BIGLASSO_OMP_H_
  int useCores = ncore < 1 ? omp_get_num_procs() : ncore;
  omp_set_dynamic(0);
  omp_set_num_threads(useCores);
#endif
}

// NOTE: in this simple function, lambda is a single value, not a path
RcppExport SEXP cdfit_gaussian_simple(SEXP X_,
                                      SEXP y_,
                                      SEXP r_,
                                      SEXP init_,
                                      SEXP xtx_,
                                      SEXP penalty_,
                                      SEXP lambda_,
                                      SEXP alpha_,
                                      SEXP gamma_,
                                      SEXP eps_,
                                      SEXP max_iter_,
                                      SEXP multiplier_,
                                      SEXP ncore_) {

  XPtr<BigMatrix> xMat(X_);
  double *y = REAL(y_);
  double *init = REAL(init_);
  double *xtx = REAL(xtx_);
  double alpha = REAL(alpha_)[0];
  double gamma = REAL(gamma_)[0];
  const char *penalty = CHAR(STRING_ELT(penalty_, 0));
  int n = xMat->nrow(); // number of observations used for fitting model
  int p = xMat->ncol();
  double eps = REAL(eps_)[0];
  int max_iter = INTEGER(max_iter_)[0];
  double *m = REAL(multiplier_);
  NumericVector lambda = NumericVector::create(REAL(lambda_)[0]);

  NumericVector resid(n); // residuals to be returned
  double *r = REAL(resid);
  for (int i = 0; i < n; i++) r[i] = REAL(r_)[i];

  set_omp_cores(INTEGER(ncore_)[0]);

  double thresh = eps * sqrt(gLoss(y, n) / n);
  NumericVector loss(1);
  IntegerVector iter(1);
  // dfmax = p: a single-lambda fit never truncates on dfmax
  arma::sp_mat beta = cdfit_gaussian_simple_core(xMat, r, init, xtx, penalty, lambda, alpha,
                                                 gamma, thresh, p, max_iter, m, n, p, loss, iter);

  NumericVector b(p); // estimated coefficients
  for (int j = 0; j < p; j++) b[j] = beta(j, 0);

  // return list:
  // - b: numeric (p x 1) vector of estimated coefficients at the supplied lambda value
  // - loss: double capturing the loss at this lambda value with these coefs.
  // - iter: integer capturing the number of iterations needed in the coordinate descent
  // - resid: numeric (n x 1) vector of residuals
  return List::create(b, loss[0], iter[0], resid);
}


// NOTE: in this simple function, lambda is a user-supplied path
RcppExport SEXP cdfit_gaussian_simple_path(SEXP X_,
                                           SEXP y_,
                                           SEXP r_,
                                           SEXP init_,
                                           SEXP xtx_,
                                           SEXP penalty_,
                                           SEXP lambda_,
                                           SEXP nlambda_,
                                           SEXP alpha_,
                                           SEXP gamma_,
                                           SEXP eps_,
                                           SEXP dfmax_,
                                           SEXP max_iter_,
                                           SEXP multiplier_,
                                           SEXP ncore_) {

  XPtr<BigMatrix> xMat(X_);
  double *y = REAL(y_);
  double *init = REAL(init_);
  double *xtx = REAL(xtx_);
  double alpha = REAL(alpha_)[0];
  double gamma = REAL(gamma_)[0];
  int n = xMat->nrow(); // number of observations used for fitting model
  int p = xMat->ncol();
  double eps = REAL(eps_)[0];
  int dfmax = INTEGER(dfmax_)[0];
  int max_iter = INTEGER(max_iter_)[0];
  double *m = REAL(multiplier_);
  const char *penalty = CHAR(STRING_ELT(penalty_, 0));
  NumericVector lambda = Rcpp::as<NumericVector>(lambda_);

  NumericVector resid(n); // residuals to be returned
  double *r = REAL(resid);
  for (int i = 0; i < n; i++) r[i] = REAL(r_)[i];

  set_omp_cores(INTEGER(ncore_)[0]);

  double thresh = eps * sqrt(gLoss(y, n) / n);
  int L = INTEGER(nlambda_)[0];
  NumericVector loss(L);
  IntegerVector iter(L);
  arma::sp_mat beta = cdfit_gaussian_simple_core(xMat, r, init, xtx, penalty, lambda, alpha,
                                                 gamma, thresh, dfmax, max_iter, m, n, p, loss,
                                                 iter);

  // return list:
  // - beta: numeric (p x L) sparse matrix of estimated coefficients along the lambda path
  // - loss: double capturing the loss at this lambda value with these coefs.
  // - iter: integer capturing the number of iterations needed in the coordinate descent
  // - resid: numeric (n x 1) vector of residuals
  return List::create(beta, loss, iter, resid);
}
