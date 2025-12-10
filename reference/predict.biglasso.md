# Model predictions based on a fitted `biglasso` object

Extract predictions (fitted reponse, coefficients, etc.) from a fitted
[`biglasso()`](https://pbreheny.github.io/biglasso/reference/biglasso.md)
object.

## Usage

``` r
# S3 method for class 'biglasso'
predict(
  object,
  X,
  row.idx = 1:nrow(X),
  type = c("link", "response", "class", "coefficients", "vars", "nvars"),
  lambda = NULL,
  which = 1:length(object$lambda),
  ...
)

# S3 method for class 'mbiglasso'
predict(
  object,
  X,
  row.idx = 1:nrow(X),
  type = c("link", "response", "coefficients", "vars", "nvars"),
  lambda = NULL,
  which = 1:length(object$lambda),
  k = 1,
  ...
)

# S3 method for class 'biglasso'
coef(object, lambda = NULL, which = 1:length(object$lambda), drop = TRUE, ...)

# S3 method for class 'mbiglasso'
coef(
  object,
  lambda = NULL,
  which = 1:length(object$lambda),
  intercept = TRUE,
  ...
)
```

## Arguments

- object:

  A fitted `"biglasso"` model object.

- X:

  Matrix of values at which predictions are to be made. It must be a
  [`bigmemory::big.matrix()`](https://rdrr.io/pkg/bigmemory/man/big.matrix.html)
  object. Not used for `type="coefficients"`.

- row.idx:

  Similar to that in
  [`biglasso()`](https://pbreheny.github.io/biglasso/reference/biglasso.md),
  it's a vector of the row indices of `X` that used for the prediction.
  `1:nrow(X)` by default.

- type:

  Type of prediction:

  - `"link"` returns the linear predictors

  - `"response"` gives the fitted values

  - `"class"` returns the binomial outcome with the highest probability

  - `"coefficients"` returns the coefficients

  - `"vars"` returns a list containing the indices and names of the
    nonzero variables at each value of `lambda`

  - `"nvars"` returns the number of nonzero coefficients at each value
    of `lambda`

- lambda:

  Values of the regularization parameter `lambda` at which predictions
  are requested. Linear interpolation is used for values of `lambda` not
  in the sequence of lambda values in the fitted models.

- which:

  Indices of the penalty parameter `lambda` at which predictions are
  required. By default, all indices are returned. If `lambda` is
  specified, this will override `which`.

- ...:

  Not used.

- k:

  Index of the response to predict in multiple responses regression (
  `family="mgaussian"`).

- drop:

  If coefficients for a single value of `lambda` are to be returned,
  reduce dimensions to a vector? Setting `drop=FALSE` returns a 1-column
  matrix.

- intercept:

  Whether the intercept should be included in the returned coefficients.
  For `family="mgaussian"` only.

## Value

The object returned depends on `type`.

## See also

[`biglasso()`](https://pbreheny.github.io/biglasso/reference/biglasso.md),
[`cv.biglasso()`](https://pbreheny.github.io/biglasso/reference/cv.biglasso.md)

## Author

Yaohui Zeng and Patrick Breheny

## Examples

``` r
## Logistic regression
data(colon)
x <- colon$X
y <- colon$y
x_bm <- as.big.matrix(x, backingfile = "")
fit <- biglasso(x_bm, y, penalty = "lasso", family = "binomial")
coef <- coef(fit, lambda = 0.05, drop = TRUE)
coef[which(coef != 0)]
#>  [1]  7.654998e-01 -5.099878e-05 -2.753690e-03 -4.182978e-04  5.594134e-05
#>  [6] -1.183261e-03  4.871641e-04  2.449309e-04  4.574918e-03 -2.180171e-04
#> [11] -1.093833e-03 -1.739847e-03  7.168045e-04  1.007572e-03 -2.752499e-03
#> [16]  4.657996e-03  5.308308e-03
predict(fit, x_bm, type = "link", lambda = 0.05)[1:10]
#>  [1]  0.8065784 -2.4540880  1.0150133 -0.5253085  2.0519542 -0.7598931
#>  [7]  2.2006058 -2.1774748  3.1595848 -2.7014291
predict(fit, x_bm, type = "response", lambda = 0.05)[1:10]
#>  [1] 0.69137991 0.07914011 0.73400010 0.37161178 0.88614493 0.31866947
#>  [7] 0.90030390 0.10179158 0.95928473 0.06288908
predict(fit, x_bm, type = "class", lambda = 0.1)[1:10]
#>  [1] 1 0 1 1 1 0 1 0 1 0
predict(fit, type = "vars", lambda = c(0.05, 0.1))
#> $`0.05`
#>  Hsa.8147 Hsa.36689 Hsa.42949 Hsa.22762 Hsa.692.2 Hsa.31801  Hsa.3016  Hsa.5392 
#>       249       377       617       639       765      1024      1325      1346 
#>  Hsa.1832 Hsa.12241 Hsa.44244  Hsa.2928 Hsa.41159 Hsa.33268  Hsa.6814  Hsa.1660 
#>      1423      1482      1504      1582      1641      1644      1772      1870 
#> 
#> $`0.1`
#>  Hsa.8147 Hsa.36689 Hsa.37937  Hsa.3306 Hsa.692.2  Hsa.5392  Hsa.2928 Hsa.33268 
#>       249       377       493       625       765      1346      1582      1644 
#>  Hsa.6814  Hsa.1660 
#>      1772      1870 
#> 
predict(fit, type = "nvars", lambda = c(0.05, 0.1))
#> 0.05  0.1 
#>   16   10 
```
