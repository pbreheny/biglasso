# Direct interface to biglasso fitting, no preprocessing, path version

This function is intended for users who know exactly what they're doing
and want complete control over the fitting process. It

- does NOT add an intercept

- does NOT standardize the design matrix both of the above are critical
  steps in data analysis. However, a direct API has been provided for
  use in situations where the lasso fitting process is an internal
  component of a more complicated algorithm and standardization must be
  handled externally.

## Usage

``` r
biglasso_path(
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

  A vector of numeric values the lasso tuning parameter.

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

  A sparse matrix where rows are estimates a given coefficient across
  all values of lambda

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

`biglasso_path()` works identically to
[`biglasso_fit()`](https://pbreheny.github.io/biglasso/reference/biglasso_fit.md)
except it offers the additional option of fitting models across a path
of tuning parameter values.

Note:

- Hybrid safe-strong rules are turned off for
  [`biglasso_fit()`](https://pbreheny.github.io/biglasso/reference/biglasso_fit.md),
  as these rely on standardization

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
fit <- biglasso_path(X = X.bm, y = y, r = y, init = init, xtx = xtx,
  lambda = c(0.5, 0.1, 0.05, 0.01, 0.001), 
  penalty.factor=c(0, rep(1, ncol(X)-1)), max.iter=2000)
fit$beta
#> 9 x 5 sparse Matrix of class "dgCMatrix"
#>                                                                      
#>         1.86789798  1.726547654  1.105241004  0.62236552  0.237779016
#> lcavol  0.22521668  0.577920344  0.559394365  0.55854063  0.563967682
#> lweight .           0.042348062  0.334244868  0.55897485  0.615294573
#> age     .          -0.005532399 -0.012370678 -0.01887351 -0.020994791
#> lbph    .           0.076396177  0.083246962  0.09416931  0.096496944
#> svi     .           .            0.255311017  0.63017098  0.748190547
#> lcp     .           .            .           -0.06349802 -0.101736983
#> gleason .           .            .            .           0.042445101
#> pgg45   0.01256831  0.006710624  0.005389947  0.00498825  0.004541638
  
fit <- biglasso_path(X = X.bm, y = y, r = y, init = init, xtx = xtx,
  lambda = c(0.5, 0.1, 0.05, 0.01, 0.001), penalty='MCP',
  penalty.factor=c(0, rep(1, ncol(X)-1)), max.iter = 2000)
fit$beta
#> 9 x 5 sparse Matrix of class "dgCMatrix"
#>                                                                         
#>          1.648298073 -0.115420660  0.494158018  0.187554573  0.187527451
#> lcavol   0.667634641  0.600386232  0.569551761  0.564445466  0.564444994
#> lweight  .            0.753123645  0.613981343  0.621790535  0.621791579
#> age     -0.003182706 -0.017330406 -0.020888578 -0.021242328 -0.021242354
#> lbph     .            0.007112865  0.097369534  0.096738824  0.096738703
#> svi      .            .            0.752459039  0.761518580  0.761519278
#> lcp      .            .           -0.104951529 -0.106015242 -0.106015402
#> gleason  .            .            .            0.048346251  0.048350230
#> pgg45    0.005416757  0.006411446  0.005322958  0.004471521  0.004471458
```
