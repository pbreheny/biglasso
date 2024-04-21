#' Model predictions based on a fitted [cv.biglasso()] object
#'
#' Extract predictions from a fitted [cv.biglasso()] object.
#'
#' @name predict.cv.biglasso
#' @rdname predict.cv.biglasso
#' @method predict cv.biglasso
#'
#' @param object A fitted `"cv.biglasso"` model object.
#' @param X Matrix of values at which predictions are to be made. It must be a
#' [bigmemory::big.matrix()] object. Not used for
#' `type="coefficients"`.
#' @param row.idx Similar to that in [biglasso()], it's a
#' vector of the row indices of `X` that used for the prediction.
#' `1:nrow(X)` by default.
#' @param type Type of prediction:
#'   * `"link"` returns the linear predictors
#'   * `"response"` gives the fitted values
#'   * `"class"` returns the binomial outcome with the highest probability
#'   * `"coefficients"` returns the coefficients
#'   * `"vars"` returns a list containing the indices and names of the nonzero variables at each value of `lambda`
#'   * `"nvars"` returns the number of nonzero coefficients at each value of `lambda`
#' @param lambda Values of the regularization parameter `lambda` at which
#' predictions are requested.  The default value is the one corresponding to
#' the minimum cross-validation error. Accepted values are also the strings
#' "lambda.min" (`lambda` of minimum cross-validation error) and
#' "lambda.1se" (Largest value of `lambda` for which the cross-validation
#' error was at most one standard error larger than the minimum.).
#' @param which Indices of the penalty parameter `lambda` at which
#' predictions are requested. The default value is the index of lambda
#' corresponding to lambda.min.  Note: this is overridden if `lambda` is
#' specified.
#' @param \dots Not used.
#' 
#' @returns The object returned depends on `type`.
#' 
#' @author Yaohui Zeng and Patrick Breheny
#'
#' @seealso [biglasso()], [cv.biglasso()]
#' 
#' @examples
#' \dontrun{
#' ## predict.cv.biglasso
#' data(colon)
#' X <- colon$X
#' y <- colon$y
#' X.bm <- as.big.matrix(X, backingfile = "")
#' fit <- biglasso(X.bm, y, penalty = 'lasso', family = "binomial")
#' cvfit <- cv.biglasso(X.bm, y, penalty = 'lasso', family = "binomial", seed = 1234, ncores = 2)
#' coef <- coef(cvfit)
#' coef[which(coef != 0)]
#' predict(cvfit, X.bm, type = "response")
#' predict(cvfit, X.bm, type = "link")
#' predict(cvfit, X.bm, type = "class")
#' predict(cvfit, X.bm, lambda = "lambda.1se")
#' }
#' @export

predict.cv.biglasso <- function(object, X, row.idx = 1:nrow(X),
                                type = c("link","response","class",
                                         "coefficients","vars","nvars"),
                                lambda = object$lambda.min,
                                which = object$min, ...) {
  if (is.character(lambda)) {
    lambda <- match.arg(lambda, c("lambda.min", "lambda.1se"))
    lambda <- object[[lambda]]
  }
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
