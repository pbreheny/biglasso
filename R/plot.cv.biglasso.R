#' Plots the cross-validation curve from a "cv.biglasso" object
#' 
#' Plot the cross-validation curve from a \code{\link{cv.biglasso}} object,
#' along with standard error bars.
#' 
#' Error bars representing approximate 68\% confidence intervals are plotted
#' along with the estimates at value of \code{lambda}.  For \code{rsq} and
#' \code{snr}, these confidence intervals are quite crude, especially near.
#' 
#' @param x A \code{"cv.biglasso"} object.
#' @param log.l Should horizontal axis be on the log scale?  Default is TRUE.
#' @param type What to plot on the vertical axis.  \code{cve} plots the
#' cross-validation error (deviance); \code{rsq} plots an estimate of the
#' fraction of the deviance explained by the model (R-squared); \code{snr}
#' plots an estimate of the signal-to-noise ratio; \code{scale} plots, for
#' \code{family="gaussian"}, an estimate of the scale parameter (standard
#' deviation); \code{pred} plots, for \code{family="binomial"}, the estimated
#' prediction error; \code{all} produces all of the above.
#' @param selected If \code{TRUE} (the default), places an axis on top of the
#' plot denoting the number of variables in the model (i.e., that have a
#' nonzero regression coefficient) at that value of \code{lambda}.
#' @param vertical.line If \code{TRUE} (the default), draws a vertical line at
#' the value where cross-validaton error is minimized.
#' @param col Controls the color of the dots (CV estimates).
#' @param \dots Other graphical parameters to \code{plot}
#' @author Yaohui Zeng and Patrick Breheny
#' 
#' Maintainer: Yaohui Zeng <yaohui.zeng@@gmail.com>
#' @seealso \code{\link{biglasso}}, \code{\link{cv.biglasso}}
#' @keywords models regression
#' @examples
#' 
#' ## See examples in "cv.biglasso"
#' 
#' @export
#' 
plot.cv.biglasso <- function(x, log.l = TRUE, type = c("cve", "rsq", "scale", 
                                                       "snr", "pred", "all"), 
                             selected = TRUE, vertical.line = TRUE, col = "red", ...) {
  # inherits cv.ncvreg
  class(x) <- 'cv.ncvreg'
  plot(x = x, log.l = log.l, type = type, selected = selected, 
       vertical.line = vertical.line, col = col, ...)
}
