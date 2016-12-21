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
