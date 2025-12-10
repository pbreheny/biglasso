# Plots the cross-validation curve from a "cv.biglasso" object

Plot the cross-validation curve from a
[`cv.biglasso()`](https://pbreheny.github.io/biglasso/reference/cv.biglasso.md)
object, along with standard error bars.

## Usage

``` r
# S3 method for class 'cv.biglasso'
plot(
  x,
  log.l = TRUE,
  type = c("cve", "rsq", "scale", "snr", "pred", "all"),
  selected = TRUE,
  vertical.line = TRUE,
  col = "red",
  ...
)
```

## Arguments

- x:

  A `"cv.biglasso"` object.

- log.l:

  Should horizontal axis be on the log scale? Default is TRUE.

- type:

  What to plot on the vertical axis. `cve` plots the cross-validation
  error (deviance); `rsq` plots an estimate of the fraction of the
  deviance explained by the model (R-squared); `snr` plots an estimate
  of the signal-to-noise ratio; `scale` plots, for `family="gaussian"`,
  an estimate of the scale parameter (standard deviation); `pred` plots,
  for `family="binomial"`, the estimated prediction error; `all`
  produces all of the above.

- selected:

  If `TRUE` (the default), places an axis on top of the plot denoting
  the number of variables in the model (i.e., that have a nonzero
  regression coefficient) at that value of `lambda`.

- vertical.line:

  If `TRUE` (the default), draws a vertical line at the value where
  cross-validaton error is minimized.

- col:

  Controls the color of the dots (CV estimates).

- ...:

  Other graphical parameters to `plot`

## Details

Error bars representing approximate 68\\ along with the estimates at
value of `lambda`. For `rsq` and `snr`, these confidence intervals are
quite crude, especially near.

## See also

[`biglasso()`](https://pbreheny.github.io/biglasso/reference/biglasso.md),
[`cv.biglasso()`](https://pbreheny.github.io/biglasso/reference/cv.biglasso.md)

## Author

Yaohui Zeng and Patrick Breheny

## Examples

``` r
## See examples in "cv.biglasso"
```
