# Plot coefficients from a "biglasso" object

Produce a plot of the coefficient paths for a fitted
[`biglasso()`](https://pbreheny.github.io/biglasso/reference/biglasso.md)
object.

## Usage

``` r
# S3 method for class 'biglasso'
plot(x, alpha = 1, log.l = TRUE, ...)
```

## Arguments

- x:

  Fitted
  [`biglasso()`](https://pbreheny.github.io/biglasso/reference/biglasso.md)
  model.

- alpha:

  Controls alpha-blending, helpful when the number of covariates is
  large. Default is alpha=1.

- log.l:

  Should horizontal axis be on the log scale? Default is TRUE.

- ...:

  Other graphical parameters to
  [`plot()`](https://rdrr.io/r/graphics/plot.default.html)

## See also

[`biglasso()`](https://pbreheny.github.io/biglasso/reference/biglasso.md),
[`cv.biglasso()`](https://pbreheny.github.io/biglasso/reference/cv.biglasso.md)

## Author

Yaohui Zeng and Patrick Breheny

## Examples

``` r
## See examples in "biglasso"
```
