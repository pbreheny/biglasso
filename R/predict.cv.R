#' Model predictions based on a fitted \code{\link{cv.biglasso}} object
#' 
#' Extract predictions from a fitted \code{\link{cv.biglasso}} object.
#' 
#' @name predict.cv.biglasso
#' @rdname predict.cv.biglasso
#' @method predict cv.biglasso
#' 
#' @param object A fitted \code{"cv.biglasso"} model object.
#' @param X Matrix of values at which predictions are to be made. It must be a
#' \code{\link[bigmemory]{big.matrix}} object. Not used for
#' \code{type="coefficients"}.
#' @param row.idx Similar to that in \code{\link[biglasso]{biglasso}}, it's a
#' vector of the row indices of \code{X} that used for the prediction.
#' \code{1:nrow(X)} by default.
#' @param type Type of prediction: \code{"link"} returns the linear predictors;
#' \code{"response"} gives the fitted values; \code{"class"} returns the
#' binomial outcome with the highest probability; \code{"coefficients"} returns
#' the coefficients; \code{"vars"} returns a list containing the indices and
#' names of the nonzero variables at each value of \code{lambda};
#' \code{"nvars"} returns the number of nonzero coefficients at each value of
#' \code{lambda}.
#' @param lambda Values of the regularization parameter \code{lambda} at which
#' predictions are requested.  The default value is the one corresponding to
#' the minimum cross-validation error.
#' @param which Indices of the penalty parameter \code{lambda} at which
#' predictions are requested. The default value is the index of lambda
#' corresponding to lambda.min.  Note: this is overridden if \code{lambda} is
#' specified.
#' @param \dots Not used.
#' @return The object returned depends on \code{type}.
#' @author Yaohui Zeng and Patrick Breheny
#' 
#' Maintainer: Yaohui Zeng <yaohui-zeng@@uiowa.edu>
#' @seealso \code{\link{biglasso}}, \code{\link{cv.biglasso}}
#' @keywords models regression
#' @examples
#' 
#' ## predict.cv.biglasso
#' data(colon)
#' X <- colon$X
#' y <- colon$y
#' X.bm <- as.big.matrix(X)
#' fit <- biglasso(X.bm, y, penalty = 'lasso', family = "binomial")
#' cvfit <- cv.biglasso(X.bm, y, penalty = 'lasso', family = "binomial", seed = 1234, ncores = 2)
#' coef <- coef(cvfit)
#' coef[which(coef != 0)]
#' predict(cvfit, X.bm, type = "response")
#' predict(cvfit, X.bm, type = "link")
#' predict(cvfit, X.bm, type = "class")
#'
#' @export
#' 
predict.cv.biglasso <- function(object, X, row.idx = 1:nrow(X),
                                type = c("link","response","class",
                                         "coefficients","vars","nvars"), 
                                lambda = object$lambda.min,
                                which = object$min, ...) {
  type <- match.arg(type)
  predict.biglasso(object$fit, X = X, row.idx = row.idx, type = type, 
                   lambda = lambda, which = which, ...)
}

#' @method coef cv.biglasso
#' @rdname predict.cv.biglasso
#' @export
#' 
coef.cv.biglasso <- function(object, lambda = object$lambda.min, which = object$min, ...) {
  coef.biglasso(object$fit, lambda = lambda, which = which, ...)
}
