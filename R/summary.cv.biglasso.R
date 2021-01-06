#' Summarizing inferences based on cross-validation
#' 
#' Summary method for \code{cv.biglasso} objects.
#' 
#' @name summary.cv.biglasso
#' @rdname summary.cv.biglasso
#' @method summary cv.biglasso
#' 
#' @param object A \code{cv.biglasso} object.
#' @param x A \code{"summary.cv.biglasso"} object.
#' @param digits Number of digits past the decimal point to print out.  Can be
#' a vector specifying different display digits for each of the five
#' non-integer printed values.
#' @param \dots Further arguments passed to or from other methods.
#' @return \code{summary.cv.biglasso} produces an object with S3 class
#' \code{"summary.cv.biglasso"} which inherits class
#' \code{"summary.cv.ncvreg"}.  The class has its own print method and contains
#' the following list elements: \item{penalty}{The penalty used by
#' \code{biglasso}.} \item{model}{Either \code{"linear"} or \code{"logistic"},
#' depending on the \code{family} option in \code{biglasso}.} \item{n}{Number
#' of observations} \item{p}{Number of regression coefficients (not including
#' the intercept).} \item{min}{The index of \code{lambda} with the smallest
#' cross-validation error.} \item{lambda}{The sequence of \code{lambda} values
#' used by \code{cv.biglasso}.} \item{cve}{Cross-validation error (deviance).}
#' \item{r.squared}{Proportion of variance explained by the model, as estimated
#' by cross-validation.} \item{snr}{Signal to noise ratio, as estimated by
#' cross-validation.} \item{sigma}{For linear regression models, the scale
#' parameter estimate.} \item{pe}{For logistic regression models, the
#' prediction error (misclassification error).}
#' @author Yaohui Zeng and Patrick Breheny
#' 
#' Maintainer: Yaohui Zeng <yaohui.zeng@@gmail.com>
#' @seealso \code{\link{biglasso}}, \code{\link{cv.biglasso}},
#' \code{\link{plot.cv.biglasso}}
#' @keywords models regression
#' @examples
#' ## See examples in "cv.biglasso" and "biglasso-package"
#' @export
#' 
#' 
summary.cv.biglasso <- function(object, ...) {
  # inherits cv.ncvreg
  class(object) <- 'cv.ncvreg'
  summary(object = object, ...)
}

#' @method print summary.cv.biglasso
#' @rdname summary.cv.biglasso
#' @export
#' 
print.summary.cv.biglasso <- function(x, digits, ...) {
  # inherits summary.cv.ncvreg
  class(x) <- 'summary.cv.ncvreg'
  print(x = x, ...)
}
