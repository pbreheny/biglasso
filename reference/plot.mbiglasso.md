# Plot coefficients from a "mbiglasso" object

Produce a plot of the coefficient paths for a fitted multiple responses
`mbiglasso` object.

## Usage

``` r
# S3 method for class 'mbiglasso'
plot(x, alpha = 1, log.l = TRUE, norm.beta = TRUE, ...)
```

## Arguments

- x:

  Fitted `mbiglasso` model.

- alpha:

  Controls alpha-blending, helpful when the number of covariates is
  large. Default is alpha=1.

- log.l:

  Should horizontal axis be on the log scale? Default is TRUE.

- norm.beta:

  Should the vertical axis be the l2 norm of coefficients for each
  variable? Default is TRUE. If False, the vertical axis is the
  coefficients.

- ...:

  Other graphical parameters to
  [`plot()`](https://rdrr.io/r/graphics/plot.default.html)

## See also

[`biglasso()`](https://pbreheny.github.io/biglasso/reference/biglasso.md)

## Author

Chuyi Wang

## Examples

``` r
## See examples in "biglasso"
```
