#' Plot coefficients from a "mbiglasso" object
#' 
#' Produce a plot of the coefficient paths for a fitted multiple responses
#' \code{mbiglasso} object.
#' 
#' 
#' @param x Fitted \code{"mbiglasso"} model.
#' @param alpha Controls alpha-blending, helpful when the number of covariates
#' is large.  Default is alpha=1.
#' @param log.l Should horizontal axis be on the log scale?  Default is TRUE.
#' @param norm.beta Should the vertical axis be the l2 norm of coefficients for each variable?
#' Default is TRUE. If False, the vertical axis is the coefficients.
#' @param \dots Other graphical parameters to \code{plot}
#' @author Chuyi Wang
#' 
#' Maintainer: Chuyi Wang <wwaa0208@@gmail.com>
#' @seealso \code{\link{biglasso}},
#' @keywords models regression
#' @examples
#' 
#' ## See examples in "biglasso"
#' 
#' @export
#' 
plot.mbiglasso <- function(x, alpha = 1, log.l = TRUE, norm.beta = TRUE, ...) {
  YY <- coef(x, intercept = FALSE) 
  ## currently not support unpenalized coefficients. NOT USED
  penalized <- which(x$penalty.factor!=0)
  nonzero <- which(apply(abs(YY[[1]]), 1, sum)!=0)
  ind <- intersect(penalized, nonzero)
  nclass = length(YY)
  if(norm.beta) {
    Y <- matrix(0, length(ind), length(x$lambda))
    #for(i in 1:length(ind)) {
    #  for(j in 1:length(x$lambda)) {
    #    for(class in 1:nclass) {
    #      Y[i,j] = Y[i,j] + (YY[[class]])[ind[i],j]^2
    #    }
    #  }
    #}
    for(class in 1:nclass) Y <- Y + (YY[[class]])[ind,]^2
    Y = sqrt(Y)
  } else {
    Y <- matrix(0, length(ind)*nclass, length(x$lambda))
    for(i in 1:length(ind)) {
      for(class in 1:nclass) {
        Y[(i-1)*nclass+class,] = (YY[[class]])[ind[i],]
      }
    }
  }
  p <- nrow(Y)
  l <- x$lambda
  
  if (log.l) {
    l <- log(l)
    xlab <- expression(log(lambda))
  } else {
    xlab <- expression(lambda)
  }
  plot.args <- list(x=l, y=1:length(l), ylim=range(Y), xlab=xlab, ylab="", type="n", xlim=rev(range(l)), las=1)
  new.args <- list(...)
  if (length(new.args)) {
    plot.args[names(new.args)] <- new.args
  }
  do.call("plot", plot.args)
  if (!is.element("ylab", names(new.args))) { 
    if(norm.beta) mtext(expression("||"*beta*"||"[2]), side=2, cex=par("cex"), line=2.5, las=1)
    else mtext(expression(hat(beta)), side=2, cex=par("cex"), line=3, las=1)
  }
  
  cols <- hcl(h=seq(15, 375, len=max(4, p+1)), l=60, c=150, alpha=alpha)
  cols <- if (p==2) cols[c(1,3)] else cols[1:p]  
  line.args <- list(col=cols, lwd=1+2*exp(-p/20), lty=1)
  if (length(new.args)) {
    line.args[names(new.args)] <- new.args
  }
  line.args$x <- l
  line.args$y <- t(as.matrix(Y))
  do.call("matlines",line.args)
  
  abline(h=0)
}

