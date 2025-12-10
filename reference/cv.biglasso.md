# Cross-validation for biglasso

Perform k-fold cross validation for penalized regression models over a
grid of values for the regularization parameter lambda.

## Usage

``` r
cv.biglasso(
  X,
  y,
  row.idx = 1:nrow(X),
  family = c("gaussian", "binomial", "cox", "mgaussian"),
  eval.metric = c("default", "MAPE", "auc", "class"),
  ncores = parallel::detectCores(),
  ...,
  nfolds = 5,
  seed,
  cv.ind,
  trace = FALSE,
  grouped = TRUE
)
```

## Arguments

- X:

  The design matrix, without an intercept, as in
  [`biglasso()`](https://pbreheny.github.io/biglasso/reference/biglasso.md).

- y:

  The response vector, as in `biglasso`.

- row.idx:

  The integer vector of row indices of `X` that used for fitting the
  model. as in `biglasso`.

- family:

  Either `"gaussian"`, `"binomial"`, `"cox"` or `"mgaussian"` depending
  on the response. `"cox"` and `"mgaussian"` are not supported yet.

- eval.metric:

  The evaluation metric for the cross-validated error and for choosing
  optimal `lambda`. "default" for linear regression is MSE (mean squared
  error), for logistic regression is binomial deviance. "MAPE", for
  linear regression only, is the Mean Absolute Percentage Error. "auc",
  for binary classification, is the area under the receiver operating
  characteristic curve (ROC). "class", for binary classification, gives
  the misclassification error.

- ncores:

  The number of cores to use for parallel execution of the
  cross-validation folds, run on a cluster created by the `parallel`
  package. (This is also supplied to the `ncores` argument in
  [`biglasso()`](https://pbreheny.github.io/biglasso/reference/biglasso.md),
  which is the number of OpenMP threads, but only for the first call of
  [`biglasso()`](https://pbreheny.github.io/biglasso/reference/biglasso.md)
  that is run on the entire data. The individual calls of
  [`biglasso()`](https://pbreheny.github.io/biglasso/reference/biglasso.md)
  for the CV folds are run without the `ncores` argument.)

- ...:

  Additional arguments to `biglasso`.

- nfolds:

  The number of cross-validation folds. Default is 5.

- seed:

  The seed of the random number generator in order to obtain
  reproducible results.

- cv.ind:

  Which fold each observation belongs to. By default the observations
  are randomly assigned by `cv.biglasso`.

- trace:

  If set to TRUE, cv.biglasso will inform the user of its progress by
  announcing the beginning of each CV fold. Default is FALSE.

- grouped:

  Whether to calculate CV standard error (`cvse`) over CV folds
  (`TRUE`), or over all cross-validated predictions. Ignored when
  `eval.metric` is 'auc'.

## Value

An object with S3 class `"cv.biglasso"` which inherits from class
`"cv.ncvreg"`. The following variables are contained in the class
(adopted from
[`ncvreg::cv.ncvreg()`](https://pbreheny.github.io/ncvreg/reference/cv.ncvreg.html)).

- cve:

  The error for each value of `lambda`, averaged across the
  cross-validation folds.

- cvse:

  The estimated standard error associated with each value of for `cve`.

- lambda:

  The sequence of regularization parameter values along which the
  cross-validation error was calculated.

- fit:

  The fitted `biglasso` object for the whole data.

- min:

  The index of `lambda` corresponding to `lambda.min`.

- lambda.min:

  The value of `lambda` with the minimum cross-validation error.

- lambda.1se:

  The largest value of `lambda` for which the cross-validation error is
  at most one standard error larger than the minimum cross-validation
  error.

- null.dev:

  The deviance for the intercept-only model.

- pe:

  If `family="binomial"`, the cross-validation prediction error for each
  value of `lambda`.

- cv.ind:

  Same as above.

## Details

The function calls `biglasso` `nfolds` times, each time leaving out
1/`nfolds` of the data. The cross-validation error is based on the
residual sum of squares when `family="gaussian"` and the binomial
deviance when `family="binomial"`.

The S3 class object `cv.biglasso` inherits class
[`ncvreg::cv.ncvreg()`](https://pbreheny.github.io/ncvreg/reference/cv.ncvreg.html).
So S3 functions such as `"summary", "plot"` can be directly applied to
the `cv.biglasso` object.

## See also

[`biglasso()`](https://pbreheny.github.io/biglasso/reference/biglasso.md),
[`plot.cv.biglasso()`](https://pbreheny.github.io/biglasso/reference/plot.cv.biglasso.md),
[`summary.cv.biglasso()`](https://pbreheny.github.io/biglasso/reference/summary.cv.biglasso.md),
[`setupX()`](https://pbreheny.github.io/biglasso/reference/setupX.md)

## Author

Yaohui Zeng and Patrick Breheny

## Examples

``` r
if (FALSE) { # \dontrun{
## cv.biglasso
data(colon)
X <- colon$X
y <- colon$y
X.bm <- as.big.matrix(X)

## logistic regression
cvfit <- cv.biglasso(X.bm, y, family = 'binomial', seed = 1234, ncores = 2)
par(mfrow = c(2, 2))
plot(cvfit, type = 'all')
summary(cvfit)
} # }
```
