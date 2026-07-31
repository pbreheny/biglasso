#' Plot coefficients from a "biglasso" object
#'
#' Produce a plot of the coefficient paths for a fitted [biglasso()] object.
#'
#' @param x Fitted [biglasso()] model.
#' @param alpha Controls alpha-blending, helpful when the number of covariates is large. Default is
#'   alpha=1.
#' @param log.l Should horizontal axis be on the log scale? Default is TRUE.
#' @param ... Other graphical parameters to [plot()]
#'
#' @returns Invisibly returns `NULL`. Called for its side effect of producing a plot.
#'
#' @seealso [biglasso()], [cv.biglasso()]
#'
#' @export
plot.biglasso <- function(x, alpha = 1, log.l = TRUE, ...) {
  YY <- if (length(x$penalty.factor) == nrow(x$beta)) coef(x) else coef(x)[-1, , drop = FALSE]
  ## currently not support unpenalized coefficients. NOT USED
  penalized <- which(x$penalty.factor != 0)
  nonzero <- which(apply(abs(YY), 1, sum) != 0)
  ind <- intersect(penalized, nonzero)
  Y <- as.matrix(YY[ind, , drop = FALSE]) # convert Matrix to matrix
  l <- x$lambda

  if (log.l) {
    l <- log(l)
    xlab <- expression(log(lambda))
  } else {
    xlab <- expression(lambda)
  }

  plot_biglasso_lines(
    l,
    Y,
    xlab,
    alpha,
    draw_ylab = function() {
      mtext(expression(hat(beta)), side = 2, cex = par("cex"), line = 3, las = 1)
    },
    ...
  )
  invisible(NULL)
}
