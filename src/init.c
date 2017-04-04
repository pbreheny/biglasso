#include <R.h>
#include <Rinternals.h> // for SEXP
#include <stdlib.h> //
#include <R_ext/Rdynload.h>
#include <R_ext/Visibility.h>  // optional

// Coordinate descent for logistic models
extern SEXP cdfit_binomial_hsr(SEXP X_, SEXP y_, SEXP row_idx_, 
                                   SEXP lambda_, SEXP nlambda_, SEXP lam_scale_,
                                   SEXP lambda_min_, SEXP alpha_, SEXP user_, SEXP eps_, 
                                   SEXP max_iter_, SEXP multiplier_, SEXP dfmax_, 
                                   SEXP ncore_, SEXP warn_, SEXP verbose_);

// Coordinate descent for gaussian models
extern SEXP cdfit_gaussian_hsr_dome(SEXP X_, SEXP y_, SEXP row_idx_,  
                                        SEXP lambda_, SEXP nlambda_,
                                        SEXP lam_scale_,
                                        SEXP lambda_min_, SEXP alpha_, 
                                        SEXP user_, SEXP eps_,
                                        SEXP max_iter_, SEXP multiplier_, 
                                        SEXP dfmax_, SEXP ncore_, 
                                        SEXP dome_thresh_,
                                        SEXP verbose_);

// Coordinate descent for gaussian models
extern SEXP cdfit_gaussian_hsr(SEXP X_, SEXP y_, SEXP row_idx_, 
                                   SEXP lambda_, SEXP nlambda_, 
                                   SEXP lam_scale_, SEXP lambda_min_, 
                                   SEXP alpha_, SEXP user_, SEXP eps_, 
                                   SEXP max_iter_, SEXP multiplier_, SEXP dfmax_, 
                                   SEXP ncore_, SEXP verbose_);

// Coordinate descent for gaussian models
extern SEXP cdfit_gaussian(SEXP X_, SEXP y_, SEXP row_idx_, 
                               SEXP lambda_, SEXP nlambda_, 
                               SEXP lam_scale_, SEXP lambda_min_, 
                               SEXP alpha_, SEXP user_, SEXP eps_, 
                               SEXP max_iter_, SEXP multiplier_, SEXP dfmax_, 
                               SEXP ncore_, SEXP verbose_);

// Coordinate descent for gaussian models
extern SEXP cdfit_gaussian_edpp_active(SEXP X_, SEXP y_, SEXP row_idx_, SEXP lambda_, 
                                           SEXP nlambda_, SEXP lam_scale_,
                                           SEXP lambda_min_, SEXP alpha_, 
                                           SEXP user_, SEXP eps_, SEXP max_iter_, 
                                           SEXP multiplier_, SEXP dfmax_, SEXP ncore_);

// Coordinate descent for gaussian models
extern SEXP cdfit_gaussian_edpp(SEXP X_, SEXP y_, SEXP row_idx_, SEXP lambda_, 
                                    SEXP nlambda_, SEXP lam_scale_,
                                    SEXP lambda_min_, SEXP alpha_, 
                                    SEXP user_, SEXP eps_, SEXP max_iter_, 
                                    SEXP multiplier_, SEXP dfmax_, SEXP ncore_) ;

// Coordinate descent for gaussian models
extern SEXP cdfit_gaussian_hsr_bedpp_nac(SEXP X_, SEXP y_, SEXP row_idx_,  
                                             SEXP lambda_, SEXP nlambda_,
                                             SEXP lam_scale_,
                                             SEXP lambda_min_, SEXP alpha_, 
                                             SEXP user_, SEXP eps_,
                                             SEXP max_iter_, SEXP multiplier_, 
                                             SEXP dfmax_, SEXP ncore_, 
                                             SEXP safe_thresh_,
                                             SEXP verbose_);

// Coordinate descent for gaussian models (no active cycling)
extern SEXP cdfit_gaussian_hsr_dome_nac(SEXP X_, SEXP y_, SEXP row_idx_,  
                                            SEXP lambda_, SEXP nlambda_,
                                            SEXP lam_scale_,
                                            SEXP lambda_min_, SEXP alpha_, 
                                            SEXP user_, SEXP eps_,
                                            SEXP max_iter_, SEXP multiplier_, 
                                            SEXP dfmax_, SEXP ncore_, 
                                            SEXP dome_thresh_,
                                            SEXP verbose_);

// Coordinate descent for logistic models
extern SEXP cdfit_binomial_hsr_approx(SEXP X_, SEXP y_, SEXP row_idx_, 
                                          SEXP lambda_, SEXP nlambda_,
                                          SEXP lambda_min_, SEXP alpha_, 
                                          SEXP user_, SEXP eps_, 
                                          SEXP max_iter_, SEXP multiplier_, SEXP dfmax_, 
                                          SEXP ncore_, SEXP warn_, SEXP verbose_);

// Coordinate descent for logistic models
extern SEXP cdfit_binomial_hsr_slores(SEXP X_, SEXP y_, SEXP n_pos_, SEXP ylab_, 
                                          SEXP row_idx_, SEXP lambda_, SEXP nlambda_,
                                          SEXP lam_scale_, SEXP lambda_min_, 
                                          SEXP alpha_, SEXP user_, 
                                          SEXP eps_, SEXP max_iter_, SEXP multiplier_, 
                                          SEXP dfmax_, SEXP ncore_, SEXP warn_,
                                          SEXP safe_thresh_, SEXP verbose_);

// Coordinate descent for gaussian models (no active cycling)
extern SEXP cdfit_gaussian_hsr_nac(SEXP X_, SEXP y_, SEXP row_idx_, 
                                       SEXP lambda_, SEXP nlambda_, 
                                       SEXP lam_scale_, SEXP lambda_min_, 
                                       SEXP alpha_, SEXP user_, SEXP eps_, 
                                       SEXP max_iter_, SEXP multiplier_, SEXP dfmax_, 
                                       SEXP ncore_, SEXP verbose_);

// Coordinate descent for gaussian models (no active cycling)
extern SEXP cdfit_gaussian_nac(SEXP X_, SEXP y_, SEXP row_idx_, 
                                   SEXP lambda_, SEXP nlambda_, 
                                   SEXP lam_scale_, SEXP lambda_min_, 
                                   SEXP alpha_, SEXP user_, SEXP eps_, 
                                   SEXP max_iter_, SEXP multiplier_, SEXP dfmax_, 
                                   SEXP ncore_, SEXP verbose_);

extern SEXP biglasso_get_eta(SEXP xPSEXP, SEXP row_idx_SEXP, SEXP betaSEXP, SEXP idx_pSEXP, SEXP idx_lSEXP);

static R_CallMethodDef callMethods[] = {
  {"cdfit_binomial_hsr", (DL_FUNC) &cdfit_binomial_hsr, 16},
  {"cdfit_gaussian_hsr_dome", (DL_FUNC) &cdfit_gaussian_hsr_dome, 16},
  {"cdfit_gaussian_hsr", (DL_FUNC) &cdfit_gaussian_hsr, 15},
  {"cdfit_gaussian", (DL_FUNC) &cdfit_gaussian, 15},
  {"cdfit_gaussian_edpp_active", (DL_FUNC) &cdfit_gaussian_edpp_active, 14},
  {"cdfit_gaussian_edpp", (DL_FUNC) &cdfit_gaussian_edpp, 14},
  {"cdfit_gaussian_hsr_bedpp_nac", (DL_FUNC) &cdfit_gaussian_hsr_bedpp_nac, 16},
  {"cdfit_gaussian_hsr_dome_nac", (DL_FUNC) &cdfit_gaussian_hsr_dome_nac, 16},
  {"cdfit_binomial_hsr_approx", (DL_FUNC) &cdfit_binomial_hsr_approx, 15},
  {"cdfit_binomial_hsr_slores", (DL_FUNC) &cdfit_binomial_hsr_slores, 19},
  {"cdfit_gaussian_hsr_nac", (DL_FUNC) &cdfit_gaussian_hsr_nac, 15},
  {"cdfit_gaussian_nac", (DL_FUNC) &cdfit_gaussian_nac, 15},
  {"biglasso_get_eta", (DL_FUNC) &biglasso_get_eta, 5},
  {NULL, NULL, 0}
};

void R_init_biglasso(DllInfo *dll) {
  R_registerRoutines(dll,NULL,callMethods,NULL,NULL);
  R_useDynamicSymbols(dll, FALSE);
}
