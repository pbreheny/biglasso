# Model predictions based on a fitted [`cv.biglasso()`](https://pbreheny.github.io/biglasso/reference/cv.biglasso.md) object

Extract predictions from a fitted
[`cv.biglasso()`](https://pbreheny.github.io/biglasso/reference/cv.biglasso.md)
object.

## Usage

``` r
# S3 method for class 'cv.biglasso'
predict(
  object,
  X,
  row.idx = 1:nrow(X),
  type = c("link", "response", "class", "coefficients", "vars", "nvars"),
  lambda = object$lambda.min,
  which = object$min,
  ...
)

# S3 method for class 'cv.biglasso'
coef(object, lambda = object$lambda.min, which = object$min, ...)
```

## Arguments

- object:

  A fitted `"cv.biglasso"` model object.

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
  are requested. The default value is the one corresponding to the
  minimum cross-validation error. Accepted values are also the strings
  "lambda.min" (`lambda` of minimum cross-validation error) and
  "lambda.1se" (Largest value of `lambda` for which the cross-validation
  error was at most one standard error larger than the minimum.).

- which:

  Indices of the penalty parameter `lambda` at which predictions are
  requested. The default value is the index of lambda corresponding to
  lambda.min. Note: this is overridden if `lambda` is specified.

- ...:

  Not used.

## Value

The object returned depends on `type`.

## See also

[`biglasso()`](https://pbreheny.github.io/biglasso/reference/biglasso.md),
[`cv.biglasso()`](https://pbreheny.github.io/biglasso/reference/cv.biglasso.md)

## Author

Yaohui Zeng and Patrick Breheny

## Examples

``` r
if (FALSE) { # \dontrun{
## predict.cv.biglasso
data(colon)
X <- colon$X
y <- colon$y
X.bm <- as.big.matrix(X, backingfile = "")
fit <- biglasso(X.bm, y, penalty = 'lasso', family = "binomial")
cvfit <- cv.biglasso(X.bm, y, penalty = 'lasso', family = "binomial", seed = 1234, ncores = 2)
coef <- coef(cvfit)
coef[which(coef != 0)]
predict(cvfit, X.bm, type = "response")
predict(cvfit, X.bm, type = "link")
predict(cvfit, X.bm, type = "class")
predict(cvfit, X.bm, lambda = "lambda.1se")
} # }
```
