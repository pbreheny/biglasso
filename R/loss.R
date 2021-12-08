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
#' "auc", for logistic regression, is the area under the receiver operating
#' characteristic curve (ROC).
#' @param grouped Whether to calculate loss for the entire CV fold
#' (\code{TRUE}), or for predictions individually. Must be \code{TRUE} when
#' \code{eval.metric} is 'auc'.
#' @author Yaohui Zeng and Patrick Breheny
#'
#' Maintainer: Yaohui Zeng <yaohui.zeng@@gmail.com>
#' @keywords internal
#'
loss.biglasso <- function(y, yhat, family, eval.metric, grouped = TRUE) {
  n <- length(y)
  if (!is.matrix(yhat)) {
    yhat <- as.matrix(yhat)
  }
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
      val <- matrix(NA, nrow=nrow(yhat), ncol=ncol(yhat))
      if (sum(y==1)) val[y==1,] <- -2*log(yhat[y==1, , drop=FALSE])
      if (sum(y==0)) val[y==0,] <- -2*log(1-yhat[y==0, , drop=FALSE])
    } else if (eval.metric == 'auc') {
      if (!grouped) stop("eval.metric == 'auc' needs grouped == TRUE")
      grouped <- FALSE  # ironically! Because grouping won't be necessary in the end.
      # AUC estimator according to e.g. 10.1007/978-3-540-74976-9_8:
      # D0: indices of negative examples
      # D1: indices of positive examples
      # (1) > auc <- 0
      # (2) > for (t0 in D0) for (t1 in D1)
      # (3) >   auc <- auc + (yhat[t0] < yhat[t1])
      # (4) > auc <- auc / (length(D0) * length(D1))
      # Simplify (3):
      # (3) > auc <- auc + sum(yhat[D0] < yhat[t1])
      # ...
      # (3) > auc <- auc + sum(yhat < yhat[t1]) - sum(yhat[D1] < yhat[t1])
      # ...
      # (3) > auc <- auc + rank(yhat)[t1] - length(D1) - sum(yhat[D1] < yhat[t1])
      # Recognize that 'for (t1 in D1) auc <- auc + rank(yhat)[t1]' vectorizes,
      # and that 'for (t1 in D1) auc <- auc - length(D1) - sum(yhat[D1]<yhat[t1])'
      # is 'auc - length(D1) * (length(D1) + 1) / 2'.
      # This leads to the expression
      # > auc <- (sum(rank(yhat)[t1]) - length(D1) * (length(D1) + 1) / 2) /
      # >        (length(D0) * length(D1))
      # Which further simplifies to:
      # > auc <- (mean(rank(yhat)[t1]) - (length(D1) + 1) / 2) / length(D0)
      # In case of ties we calculate the expected value over randomly broken
      # ties, which corresponds to 'ties.method = "average"'.
      t1 <- y == 1
      t1.len <- sum(t1)
      t0.len <- n - t1.len
      if (t0.len == 0 || t1.len == 0) {
        # This happens when we have a bad CV split.
        # In this case the AUC is not defined. We return NA to ignore this fold.
        val <- rep(NA_real_, ncol(yhat))
      } else {
        val <- apply(yhat, 2, function(yhcol) {
          (mean(rank(yhcol, ties.method = "average")[t1]) - (t1.len + 1) / 2) /
            t0.len
        })
      }
    } else if (eval.metric == 'class') {
      yhat.class <- yhat > 0.5
      val <- yhat.class != y
      mode(val) <- "numeric"
    } else {
      stop(sprintf("eval.metric %s not supported for family %s.", eval.metric, family))
    }
  } else if (family=="poisson") {
    yly <- y*log(y)
    yly[y==0] <- 0
    val <- 2*(yly - y + yhat - y*log(yhat))
  }
  if (grouped) {
    val <- apply(val, 2, mean)
  }
  val
}
