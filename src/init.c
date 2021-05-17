#include <R.h>
#include <Rinternals.h> // for SEXP
#include <stdlib.h> //
#include <R_ext/Rdynload.h>
#include <R_ext/Visibility.h>  // optional


// Coordinate descent for mgaussian model
extern SEXP cdfit_mgaussian_ssr(SEXP X_, SEXP y_, SEXP row_idx_, 
                               SEXP lambda_, SEXP nlambda_, 
                               SEXP lam_scale_, SEXP lambda_min_, 
                               SEXP alpha_, SEXP user_, SEXP eps_, 
                               SEXP max_iter_, SEXP multiplier_, SEXP dfmax_, 
                               SEXP ncore_, SEXP safe_thresh_, 
                               SEXP update_thresh_, SEXP verbose_);

extern SEXP cdfit_mgaussian_ada(SEXP X_, SEXP y_, SEXP row_idx_, 
                                SEXP lambda_, SEXP nlambda_, 
                                SEXP lam_scale_, SEXP lambda_min_, 
                                SEXP alpha_, SEXP user_, SEXP eps_, 
                                SEXP max_iter_, SEXP multiplier_, SEXP dfmax_, 
                                SEXP ncore_, SEXP verbose_);

// Coordinate descent for cox model
extern SEXP cdfit_cox(SEXP X_, SEXP y_, SEXP d_, SEXP d_idx_, SEXP row_idx_, 
                      SEXP lambda_, SEXP nlambda_, SEXP lam_scale_,
                      SEXP lambda_min_, SEXP alpha_, SEXP user_, SEXP eps_, 
                      SEXP max_iter_, SEXP multiplier_, SEXP dfmax_, 
                      SEXP ncore_, SEXP warn_, SEXP verbose_);

extern SEXP cdfit_cox_ssr(SEXP X_, SEXP y_, SEXP d_, SEXP d_idx_, SEXP row_idx_, 
                          SEXP lambda_, SEXP nlambda_, SEXP lam_scale_,
                          SEXP lambda_min_, SEXP alpha_, SEXP user_, SEXP eps_, 
                          SEXP max_iter_, SEXP multiplier_, SEXP dfmax_, 
                          SEXP ncore_, SEXP warn_, SEXP verbose_);

extern SEXP cdfit_cox_scox(SEXP X_, SEXP y_, SEXP d_, SEXP d_idx_, SEXP row_idx_, 
                               SEXP lambda_, SEXP nlambda_, SEXP lam_scale_,
                               SEXP lambda_min_, SEXP alpha_, SEXP user_, SEXP eps_, 
                               SEXP max_iter_, SEXP multiplier_, SEXP dfmax_, 
                               SEXP ncore_, SEXP warn_, SEXP safe_thresh_, SEXP verbose_);

extern SEXP cdfit_cox_ada_scox(SEXP X_, SEXP y_, SEXP d_, SEXP d_idx_, SEXP row_idx_, 
                               SEXP lambda_, SEXP nlambda_, SEXP lam_scale_,
                               SEXP lambda_min_, SEXP alpha_, SEXP user_, SEXP eps_, 
                               SEXP max_iter_, SEXP multiplier_, SEXP dfmax_, SEXP ncore_,
                               SEXP warn_, SEXP safe_thresh_, SEXP update_thresh_, SEXP verbose_);

extern SEXP cdfit_cox_sscox(SEXP X_, SEXP y_, SEXP d_, SEXP d_idx_, SEXP row_idx_, 
                            SEXP lambda_, SEXP nlambda_, SEXP lam_scale_,
                            SEXP lambda_min_, SEXP alpha_, SEXP user_, SEXP eps_, 
                            SEXP max_iter_, SEXP multiplier_, SEXP dfmax_, 
                            SEXP ncore_, SEXP warn_, SEXP safe_thresh_, SEXP verbose_);

extern SEXP cdfit_cox_safe(SEXP X_, SEXP y_, SEXP d_, SEXP d_idx_, SEXP row_idx_, 
                               SEXP lambda_, SEXP nlambda_, SEXP lam_scale_,
                               SEXP lambda_min_, SEXP alpha_, SEXP user_, SEXP eps_, 
                               SEXP max_iter_, SEXP multiplier_, SEXP dfmax_, 
                               SEXP ncore_, SEXP warn_, SEXP safe_thresh_, SEXP verbose_);

// Coordinate descent for logistic models
extern SEXP cdfit_binomial_ssr(SEXP X_, SEXP y_, SEXP row_idx_, 
                               SEXP lambda_, SEXP nlambda_, SEXP lam_scale_,
                               SEXP lambda_min_, SEXP alpha_, SEXP user_, SEXP eps_, 
                               SEXP max_iter_, SEXP multiplier_, SEXP dfmax_, 
                               SEXP ncore_, SEXP warn_, SEXP verbose_);

extern SEXP cdfit_binomial_ssr_approx(SEXP X_, SEXP y_, SEXP row_idx_, 
                                      SEXP lambda_, SEXP nlambda_,
                                      SEXP lambda_min_, SEXP alpha_, 
                                      SEXP user_, SEXP eps_, 
                                      SEXP max_iter_, SEXP multiplier_, SEXP dfmax_, 
                                      SEXP ncore_, SEXP warn_, SEXP verbose_);

extern SEXP cdfit_binomial_slores_ssr(SEXP X_, SEXP y_, SEXP n_pos_, SEXP ylab_, 
                                      SEXP row_idx_, SEXP lambda_, SEXP nlambda_,
                                      SEXP lam_scale_, SEXP lambda_min_, 
                                      SEXP alpha_, SEXP user_, 
                                      SEXP eps_, SEXP max_iter_, SEXP multiplier_, 
                                      SEXP dfmax_, SEXP ncore_, SEXP warn_,
                                      SEXP safe_thresh_, SEXP verbose_);

extern SEXP cdfit_binomial_ada_slores_ssr(SEXP X_, SEXP y_, SEXP n_pos_, SEXP ylab_, 
                                          SEXP row_idx_, SEXP lambda_, SEXP nlambda_,
                                          SEXP lam_scale_, SEXP lambda_min_, 
                                          SEXP alpha_, SEXP user_, 
                                          SEXP eps_, SEXP max_iter_, SEXP multiplier_, 
                                          SEXP dfmax_, SEXP ncore_, SEXP warn_,
                                          SEXP safe_thresh_, SEXP update_thresh_, SEXP verbose_);

// Coordinate descent for gaussian models
extern SEXP cdfit_gaussian_ada_edpp_ssr(SEXP X_, SEXP y_, SEXP row_idx_, SEXP lambda_, 
                                        SEXP nlambda_, SEXP lam_scale_,
                                        SEXP lambda_min_, SEXP alpha_, 
                                        SEXP user_, SEXP eps_, SEXP max_iter_, 
                                        SEXP multiplier_, SEXP dfmax_, SEXP ncore_,
                                        SEXP update_thresh_, SEXP verbose_);

extern SEXP cdfit_gaussian_ssr(SEXP X_, SEXP y_, SEXP row_idx_, 
                               SEXP lambda_, SEXP nlambda_, 
                               SEXP lam_scale_, SEXP lambda_min_, 
                               SEXP alpha_, SEXP user_, SEXP eps_, 
                               SEXP max_iter_, SEXP multiplier_, SEXP dfmax_, 
                               SEXP ncore_, SEXP verbose_);

extern SEXP cdfit_gaussian_bedpp_ssr(SEXP X_, SEXP y_, SEXP row_idx_,  
                                     SEXP lambda_, SEXP nlambda_,
                                     SEXP lam_scale_,
                                     SEXP lambda_min_, SEXP alpha_, 
                                     SEXP user_, SEXP eps_,
                                     SEXP max_iter_, SEXP multiplier_, 
                                     SEXP dfmax_, SEXP ncore_, 
                                     SEXP safe_thresh_,
                                     SEXP verbose_);

extern SEXP _biglasso_get_eta(SEXP xPSEXP, SEXP row_idx_SEXP, SEXP betaSEXP, SEXP idx_pSEXP, SEXP idx_lSEXP);

static R_CallMethodDef callMethods[] = {
  {"cdfit_mgaussian_ssr", (DL_FUNC) &cdfit_mgaussian_ssr, 15},
  {"cdfit_mgaussian_ada", (DL_FUNC) &cdfit_mgaussian_ada, 17},
  {"cdfit_cox", (DL_FUNC) &cdfit_cox, 18},
  {"cdfit_cox_ssr", (DL_FUNC) &cdfit_cox_ssr, 18},
  {"cdfit_cox_scox", (DL_FUNC) &cdfit_cox_scox, 19},
  {"cdfit_cox_sscox", (DL_FUNC) &cdfit_cox_sscox, 19},
  {"cdfit_cox_ada_scox", (DL_FUNC) &cdfit_cox_ada_scox, 20},
  {"cdfit_cox_safe", (DL_FUNC) &cdfit_cox_safe, 19},
  {"cdfit_binomial_ssr", (DL_FUNC) &cdfit_binomial_ssr, 16},
  {"cdfit_binomial_ssr_approx", (DL_FUNC) &cdfit_binomial_ssr_approx, 15},
  {"cdfit_binomial_slores_ssr", (DL_FUNC) &cdfit_binomial_slores_ssr, 19},
  {"cdfit_binomial_ada_slores_ssr", (DL_FUNC) &cdfit_binomial_ada_slores_ssr, 20},
  {"cdfit_gaussian_ada_edpp_ssr", (DL_FUNC) &cdfit_gaussian_ada_edpp_ssr, 16},
  {"cdfit_gaussian_ssr", (DL_FUNC) &cdfit_gaussian_ssr, 15},
  {"cdfit_gaussian_bedpp_ssr", (DL_FUNC) &cdfit_gaussian_bedpp_ssr, 16},
  {"_biglasso_get_eta", (DL_FUNC) &_biglasso_get_eta, 5},
  {NULL, NULL, 0}
};

void R_init_biglasso(DllInfo *dll) {
  R_registerRoutines(dll,NULL,callMethods,NULL,NULL);
  R_useDynamicSymbols(dll, FALSE);
}
