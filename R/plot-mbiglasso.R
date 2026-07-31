#' Plot coefficients from a "mbiglasso" object
#'
#' Produce a plot of the coefficient paths for a fitted multiple responses `mbiglasso` object.
#'
#' @param x Fitted `mbiglasso` model.
#' @param alpha Controls alpha-blending, helpful when the number of covariates is large. Default is
#'   alpha=1.
#' @param log.l Should horizontal axis be on the log scale? Default is TRUE.
#' @param norm.beta Should the vertical axis be the l2 norm of coefficients for each variable?
#'   Default is TRUE. If False, the vertical axis is the coefficients.
#' @param \dots Other graphical parameters to [plot()]
#'
#' @seealso [biglasso()]
#'
#' @examples
#' ## See examples in "biglasso"
#'
#' @export
#'
#' @author Chuyi Wang
plot.mbiglasso <- function(x, alpha = 1, log.l = TRUE, norm.beta = TRUE, ...) {
  YY <- coef(x, intercept = FALSE)
  ## currently not support unpenalized coefficients. NOT USED
  penalized <- which(x$penalty.factor != 0)
  nonzero <- which(apply(abs(YY[[1]]), 1, sum) != 0)
  ind <- intersect(penalized, nonzero)
  nclass <- length(YY)
  if (norm.beta) {
    Y <- matrix(0, length(ind), length(x$lambda))
    # for(i in 1:length(ind)) {
    #  for(j in 1:length(x$lambda)) {
    #    for(class in 1:nclass) {
    #      Y[i,j] = Y[i,j] + (YY[[class]])[ind[i],j]^2
    #    }
    #  }
    # }
    for (class in 1:nclass) {
      Y <- Y + (YY[[class]])[ind, ]^2
    }
    Y <- sqrt(Y)
  } else {
    Y <- matrix(0, length(ind) * nclass, length(x$lambda))
    for (i in 1:length(ind)) {
      for (class in 1:nclass) {
        Y[(i - 1) * nclass + class, ] <- (YY[[class]])[ind[i], ]
      }
    }
  }
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
      if (norm.beta) {
        mtext(expression("||" * beta * "||"[2]), side = 2, cex = par("cex"), line = 2.5, las = 1)
      } else {
        mtext(expression(hat(beta)), side = 2, cex = par("cex"), line = 3, las = 1)
      }
    },
    ...
  )
}
