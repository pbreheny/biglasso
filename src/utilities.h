#ifndef UTILITIES_H
#define UTILITIES_H

#include <math.h>
#include <string.h>
#include <R.h>
#include <Rinternals.h>
#include <Rdefines.h>
#include <Rmath.h>

//#include "defines.h"

// -----------------------------------------------------------------------------
// C functions
// -----------------------------------------------------------------------------
double sign(double x);

double sum(double *x, int n);

// count discarded features
int sum_discard(int *discards, int p);

int checkConvergence(double *beta, double *beta_old, double eps, int l, int J);

double lasso(double z, double l1, double l2, double v);

double gLoss(double *r, int n);

// -----------------------------------------------------------------------------
// C++ functions callled inside C
// -----------------------------------------------------------------------------
// get_row
int get_row_bm(SEXP xP);

// get_col
int get_col_bm(SEXP xP);

// get X[i, j]: i-th row, j-th column element
double get_elem_bm(SEXP xP, double center_, double scale_, int i, int j);

//crossprod - given specific rows of X
double crossprod_bm(SEXP xP, double *y_, int *row_idx_, double center_, 
                    double scale_, int n_row, int j);

//crossprod_resid - given specific rows of X: separate computation
double crossprod_resid(SEXP xP, double *y_, double sumY_, int *row_idx_, 
                       double center_, double scale_, int n_row, int j);

// update residul vector if variable j enters eligible set
void update_resid(SEXP xP, double *r, double shift, int *row_idx_, 
                  double center_, double scale_, int n_row, int j);

// Sum of squares of jth column of X
double sqsum_bm(SEXP xP, int n_row, int j, int useCores);

// Weighted sum of residuals
double wsum(double *r, double *w, int n_row);

// Weighted cross product of y with jth column of x
double wcrossprod_resid(SEXP xP, double *y, double sumYW_, int *row_idx_, 
                        double center_, double scale_, double *w, int n_row, int j);

// Weighted sum of squares of jth column of X
double wsqsum_bm(SEXP xP, double *w, int *row_idx_, double center_, 
                 double scale_, int n_row, int j);

// -----------------------------------------------------------------------------
// C++ functions used for EDPP rule
// -----------------------------------------------------------------------------
void update_theta(double *theta, SEXP xP, int *row_idx_, double *center, 
                  double *scale, double *y, double *beta, double lambda, 
                  int *nzero_beta, int n, int p, int l);

// V2 - <v1, v2> / ||v1||^2_2 * V1
void update_pv2(double *pv2, double *v1, double *v2, int n);

// apply EDPP 
void edpp_screen(int *discard_beta, SEXP xP, double *o, int *row_idx, 
                 double *center, double *scale, int n, int p, double rhs);
// apply EDPP 
void edpp_screen2(int *discard_beta, SEXP xP, double *o, int *row_idx, 
                  double *center, double *scale, int n, int p, double rhs); 

#endif
