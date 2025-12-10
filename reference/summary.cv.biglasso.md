# Summarizing inferences based on cross-validation

Summary method for `cv.biglasso` objects.

## Usage

``` r
# S3 method for class 'cv.biglasso'
summary(object, ...)

# S3 method for class 'summary.cv.biglasso'
print(x, digits, ...)
```

## Arguments

- object:

  A `cv.biglasso` object.

- ...:

  Further arguments passed to or from other methods.

- x:

  A `"summary.cv.biglasso"` object.

- digits:

  Number of digits past the decimal point to print out. Can be a vector
  specifying different display digits for each of the five non-integer
  printed values.

## Value

`summary.cv.biglasso` produces an object with S3 class
`"summary.cv.biglasso"`. The class has its own print method and contains
the following list elements:

- penalty:

  The penalty used by `biglasso`.

- model:

  Either `"linear"` or `"logistic"`, depending on the `family` option in
  `biglasso`.

- n:

  Number of observations

- p:

  Number of regression coefficients (not including the intercept).

- min:

  The index of `lambda` with the smallest cross-validation error.

- lambda:

  The sequence of `lambda` values used by `cv.biglasso`.

- cve:

  Cross-validation error (deviance).

- r.squared:

  Proportion of variance explained by the model, as estimated by
  cross-validation.

- snr:

  Signal to noise ratio, as estimated by cross-validation.

- sigma:

  For linear regression models, the scale parameter estimate.

- pe:

  For logistic regression models, the prediction error
  (misclassification error).

## See also

[`biglasso()`](https://pbreheny.github.io/biglasso/reference/biglasso.md),
[`cv.biglasso()`](https://pbreheny.github.io/biglasso/reference/cv.biglasso.md),
[`plot.cv.biglasso()`](https://pbreheny.github.io/biglasso/reference/plot.cv.biglasso.md),
[biglasso-package](https://pbreheny.github.io/biglasso/reference/biglasso-package.md)

## Author

Yaohui Zeng and Patrick Breheny

## Examples

``` r
## See examples in "cv.biglasso" and "biglasso-package"
```
