# Direct interface to biglasso fitting, no preprocessing

This function is intended for users who know exactly what they're doing
and want complete control over the fitting process. It

- does NOT add an intercept

- does NOT standardize the design matrix

- does NOT set up a path for lambda (the lasso tuning parameter) all of
  the above are critical steps in data analysis. However, a direct API
  has been provided for use in situations where the lasso fitting
  process is an internal component of a more complicated algorithm and
  standardization must be handled externally.

## Usage

``` r
biglasso_fit(
  X,
  y,
  r,
  init = rep(0, ncol(X)),
  xtx,
  penalty = "lasso",
  lambda,
  alpha = 1,
  gamma,
  ncores = 1,
  max.iter = 1000,
  eps = 1e-05,
  dfmax = ncol(X) + 1,
  penalty.factor = rep(1, ncol(X)),
  warn = TRUE,
  output.time = FALSE,
  return.time = TRUE
)
```

## Arguments

- X:

  The design matrix, without an intercept. It must be a double type
  [`bigmemory::big.matrix()`](https://rdrr.io/pkg/bigmemory/man/big.matrix.html)
  object.

- y:

  The response vector

- r:

  Residuals (length n vector) corresponding to `init`. WARNING: If you
  supply an incorrect value of `r`, the solution will be incorrect.

- init:

  Initial values for beta. Default: zero (length p vector)

- xtx:

  X scales: the jth element should equal `crossprod(X[,j])/n`. In
  particular, if X is standardized, one should pass `xtx = rep(1, p)`.
  WARNING: If you supply an incorrect value of `xtx`, the solution will
  be incorrect. (length p vector)

- penalty:

  String specifying which penalty to use. Default is 'lasso', Other
  options are 'SCAD' and 'MCP' (the latter are non-convex)

- lambda:

  A single value for the lasso tuning parameter.

- alpha:

  The elastic-net mixing parameter that controls the relative
  contribution from the lasso (l1) and the ridge (l2) penalty. The
  penalty is defined as: \$\$ \alpha\|\|\beta\|\|\_1 +
  (1-\alpha)/2\|\|\beta\|\|\_2^2.\$\$ `alpha=1` is the lasso penalty,
  `alpha=0` the ridge penalty, `alpha` in between 0 and 1 is the
  elastic-net ("enet") penalty.

- gamma:

  Tuning parameter value for nonconvex penalty. Defaults are 3.7 for
  `penalty = 'SCAD'` and 3 for `penalty = 'MCP'`

- ncores:

  The number of OpenMP threads used for parallel computing.

- max.iter:

  Maximum number of iterations. Default is 1000.

- eps:

  Convergence threshold for inner coordinate descent. The algorithm
  iterates until the maximum change in the objective after any
  coefficient update is less than `eps` times the null deviance. Default
  value is `1e-7`.

- dfmax:

  Upper bound for the number of nonzero coefficients. Default is no
  upper bound. However, for large data sets, computational burden may be
  heavy for models with a large number of nonzero coefficients.

- penalty.factor:

  A multiplicative factor for the penalty applied to each coefficient.
  If supplied, `penalty.factor` must be a numeric vector of length equal
  to the number of columns of `X`.

- warn:

  Return warning messages for failures to converge and model saturation?
  Default is TRUE.

- output.time:

  Whether to print out the start and end time of the model fitting.
  Default is FALSE.

- return.time:

  Whether to return the computing time of the model fitting. Default is
  TRUE.

## Value

An object with S3 class `"biglasso"` with following variables.

- beta:

  The vector of estimated coefficients

- iter:

  A vector of length `nlambda` containing the number of iterations until
  convergence

- resid:

  Vector of residuals calculated from estimated coefficients.

- lambda:

  The sequence of regularization parameter values in the path.

- alpha:

  Same as in
  [`biglasso()`](https://pbreheny.github.io/biglasso/reference/biglasso.md)

- loss:

  A vector containing either the residual sum of squares of the fitted
  model at each value of lambda.

- penalty.factor:

  Same as in
  [`biglasso()`](https://pbreheny.github.io/biglasso/reference/biglasso.md).

- n:

  The number of observations used in the model fitting.

- y:

  The response vector used in the model fitting.

## Details

Note:

- Hybrid safe-strong rules are turned off for `biglasso_fit()`, as these
  rely on standardization

- Currently, the function only works with linear regression
  (`family = 'gaussian'`).

## Author

Tabitha Peter and Patrick Breheny

## Examples

``` r
data(Prostate)
X <- cbind(1, Prostate$X)
xtx <- apply(X, 2, crossprod)/nrow(X)
y <- Prostate$y
X.bm <- as.big.matrix(X)
init <- rep(0, ncol(X))
fit <- biglasso_fit(X = X.bm, y = y, r=y, init = init, xtx = xtx,
  lambda = 0.1, penalty.factor=c(0, rep(1, ncol(X)-1)), max.iter = 10000)
fit$beta
#>                    lcavol      lweight          age         lbph          svi 
#>  1.725303940  0.577848489  0.043409725 -0.005572137  0.076326241  0.000000000 
#>          lcp      gleason        pgg45 
#>  0.000000000  0.000000000  0.006712771 
  
fit <- biglasso_fit(X = X.bm, y = y, r=y, init = init, xtx = xtx, penalty='MCP',
  lambda = 0.1, penalty.factor=c(0, rep(1, ncol(X)-1)), max.iter = 10000)
fit$beta
#>                    lcavol      lweight          age         lbph          svi 
#>  2.268444208  0.677388754  0.000000000 -0.013317940  0.143711214  0.000000000 
#>          lcp      gleason        pgg45 
#>  0.000000000  0.000000000  0.005398707 
```
