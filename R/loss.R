#' Internal biglasso functions
#' 
#' Internal biglasso functions
#' 
#' These are not intended for use by users.\code{loss.biglasso} calculates the 
#' value of the loss function for the given predictions (used for cross-validation).
#' 
#' @param y The observed response vector. 
#' @param yhat The predicted response vector.
#' @param family Either "gaussian" or "binomial", depending on the response.
#' @param eval.metric The evaluation metric for the cross-validated error and
#' for choosing optimal \code{lambda}. "default" for linear regression is MSE
#' (mean squared error), for logistic regression is misclassification error.
#' "MAPE", for linear regression only, is the Mean Absolute Percentage Error.
#' @author Yaohui Zeng and Patrick Breheny
#' 
#' Maintainer: Yaohui Zeng <yaohui.zeng@@gmail.com>
#' @keywords internal
#'
loss.biglasso <- function(y, yhat, family, eval.metric) {
  n <- length(y)
  if (family=="gaussian") {
    if (eval.metric == 'default') {
      val <- (y-yhat)^2
    } else if (eval.metric == "MAPE") { # Mean Absolute Percent Error (MAPE)
      val <- sweep(abs(y-as.matrix(yhat)), 1, y, "/")
    } else {
      stop("Not supported")
    }
  } else if (family=="binomial") {
    if (eval.metric == 'default') {
      yhat[yhat < 0.00001] <- 0.00001
      yhat[yhat > 0.99999] <- 0.99999
      if (is.matrix(yhat)) {
        val <- matrix(NA, nrow=nrow(yhat), ncol=ncol(yhat))
        if (sum(y==1)) val[y==1,] <- -2*log(yhat[y==1, , drop=FALSE])
        if (sum(y==0)) val[y==0,] <- -2*log(1-yhat[y==0, , drop=FALSE])
      } else {
        val <- numeric(length(y))
        if (sum(y==1)) val[y==1] <- -2*log(yhat[y==1])
        if (sum(y==0)) val[y==0] <- -2*log(1-yhat[y==0])
      }
    }
  } else if (family=="poisson") {
    yly <- y*log(y)
    yly[y==0] <- 0
    val <- 2*(yly - y + yhat - y*log(yhat))
  }
  val
}
