#' Summarizing inferences based on cross-validation
#' 
#' Summary method for `cv.biglasso` objects.
#' 
#' @name summary.cv.biglasso
#' @rdname summary.cv.biglasso
#' @method summary cv.biglasso
#' 
#' @param object A `cv.biglasso` object.
#' @param x A `"summary.cv.biglasso"` object.
#' @param digits Number of digits past the decimal point to print out.  Can be
#' a vector specifying different display digits for each of the five
#' non-integer printed values.
#' @param \dots Further arguments passed to or from other methods.
#' 
#' @returns `summary.cv.biglasso` produces an object with S3 class
#' `"summary.cv.biglasso"`. The class has its own print method and contains
#' the following list elements:
#' \item{penalty}{The penalty used by `biglasso`.}
#' \item{model}{Either `"linear"` or `"logistic"`, depending on the `family` option in `biglasso`.}
#' \item{n}{Number of observations}
#' \item{p}{Number of regression coefficients (not including the intercept).}
#' \item{min}{The index of `lambda` with the smallest cross-validation error.}
#' \item{lambda}{The sequence of `lambda` values used by `cv.biglasso`.}
#' \item{cve}{Cross-validation error (deviance).}
#' \item{r.squared}{Proportion of variance explained by the model, as estimated by cross-validation.}
#' \item{snr}{Signal to noise ratio, as estimated by cross-validation.}
#' \item{sigma}{For linear regression models, the scale parameter estimate.}
#' \item{pe}{For logistic regression models, the prediction error (misclassification error).}
#' 
#' @author Yaohui Zeng and Patrick Breheny
#' 
#' @seealso [biglasso()], [cv.biglasso()], [plot.cv.biglasso()], [biglasso-package]
#' 
#' @examples
#' ## See examples in "cv.biglasso" and "biglasso-package"
#' @export

summary.cv.biglasso <- function(object, ...) {
  S <- pmax(object$null.dev - object$cve, 0)
  if (!inherits(object, 'cv.ncvsurv') && object$fit$family=="gaussian") {
    rsq <- pmin(pmax(1 - object$cve/object$null.dev, 0), 1)
  } else {
    rsq <- pmin(pmax(1 - exp(object$cve-object$null.dev), 0), 1)
  }
  snr <- rsq/(1-rsq)
  nvars <- predict(object$fit, type="nvars")
  model <- switch(object$fit$family, gaussian="linear", binomial="logistic", poisson="Poisson", cox="Cox")
  val <- list(penalty=object$fit$penalty, model=model, n=object$fit$n, p=nrow(object$fit$beta)-1, min=object$min, lambda=object$lambda, cve=object$cve, r.squared=rsq, snr=snr, nvars=nvars)
  if (object$fit$family=="gaussian") val$sigma <- sqrt(object$cve)
  if (object$fit$family=="binomial") val$pe <- object$pe
  structure(val, class="summary.cv.biglasso")
}

#' @method print summary.cv.biglasso
#' @rdname summary.cv.biglasso
#' @export
#' 
print.summary.cv.biglasso <- function(x, digits, ...) {
  digits <- if (missing(digits)) digits <- c(2, 4, 2, 2, 3) else rep(digits, length.out=5)
  cat(x$penalty, "-penalized ", x$model, " regression with n=", x$n, ", p=", x$p, "\n", sep="")
  cat("At minimum cross-validation error (lambda=", formatC(x$lambda[x$min], digits[2], format="f"), "):\n", sep="")
  cat("-------------------------------------------------\n")
  cat("  Nonzero coefficients: ", x$nvars[x$min], "\n", sep="")
  cat("  Cross-validation error (deviance): ", formatC(min(x$cve), digits[1], format="f"), "\n", sep="")
  cat("  R-squared: ", formatC(max(x$r.squared), digits[3], format="f"), "\n", sep="")
  cat("  Signal-to-noise ratio: ", formatC(max(x$snr), digits[4], format="f"), "\n", sep="")
  if (x$model == "logistic") cat("  Prediction error: ", formatC(x$pe[x$min], digits[5], format="f"), "\n", sep="")
  if (x$model == "linear") cat("  Scale estimate (sigma): ", formatC(sqrt(x$cve[x$min]), digits[5], format="f"), "\n", sep="")
}
