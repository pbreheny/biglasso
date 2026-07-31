#' Model predictions based on a fitted `biglasso` object
#'
#' Extract predictions (fitted reponse, coefficients, etc.) from a
#' fitted [biglasso()] object.
#'
#' @name predict.biglasso
#' @rdname predict.biglasso
#' @method predict biglasso
#'
#' @param object A fitted `"biglasso"` model object.
#' @param X Matrix of values at which predictions are to be made. It must be a
#' [bigmemory::big.matrix()] object. Not used for `type="coefficients"`.
#' @param row.idx Similar to that in [biglasso()], it's a
#' vector of the row indices of `X` that used for the prediction.
#' `1:nrow(X)` by default.
#' @param type Type of prediction:
#'   * `"link"` returns the linear predictors
#'   * `"response"` gives the fitted values
#'   * `"class"` returns the binomial outcome with the highest probability
#'   * `"coefficients"` returns the coefficients
#'   * `"vars"` returns a list containing the indices and names of the nonzero variables
#'     at each value of `lambda`
#'   * `"nvars"` returns the number of nonzero coefficients at each value of `lambda`
#' @param lambda Values of the regularization parameter `lambda` at which
#' predictions are requested.  Linear interpolation is used for values of
#' `lambda` not in the sequence of lambda values in the fitted models.
#' @param k Index of the response to predict in multiple responses regression (
#' `family="mgaussian"`).
#' @param which Indices of the penalty parameter `lambda` at which
#' predictions are required.  By default, all indices are returned.  If
#' `lambda` is specified, this will override `which`.
#' @param intercept Whether the intercept should be included in the returned
#' coefficients. For `family="mgaussian"` only.
#' @param drop If coefficients for a single value of `lambda` are to be
#' returned, reduce dimensions to a vector?  Setting `drop=FALSE` returns
#' a 1-column matrix.
#' @param \dots Not used.
#'
#' @returns The object returned depends on `type`.
#'
#' @author Yaohui Zeng and Patrick Breheny
#'
#' @seealso [biglasso()], [cv.biglasso()]
#'
#' @examples
#' ## Logistic regression
#' data(colon)
#' x <- colon$X
#' y <- colon$y
#' x_bm <- as.big.matrix(x, backingfile = "")
#' fit <- biglasso(x_bm, y, penalty = "lasso", family = "binomial")
#' coef <- coef(fit, lambda = 0.05, drop = TRUE)
#' coef[which(coef != 0)]
#' predict(fit, x_bm, type = "link", lambda = 0.05)[1:10]
#' predict(fit, x_bm, type = "response", lambda = 0.05)[1:10]
#' predict(fit, x_bm, type = "class", lambda = 0.1)[1:10]
#' predict(fit, type = "vars", lambda = c(0.05, 0.1))
#' predict(fit, type = "nvars", lambda = c(0.05, 0.1))
#' @export

predict.biglasso <- function(
  object,
  X,
  row.idx = 1:nrow(X),
  type = c(
    "link",
    "response",
    "class",
    "coefficients",
    "vars",
    "nvars"
  ),
  lambda = NULL,
  which = 1:length(object$lambda),
  ...
) {
  type <- match.arg(type)
  beta <- coef.biglasso(object, lambda = lambda, which = which, drop = FALSE)
  res <- predict_biglasso_common(beta, X, row.idx, type, strip_intercept = object$family != "cox")
  if (res$done) return(res$value)
  eta <- res$eta

  # Binomial case
  if (object$family == "binomial") {
    out <- switch(type,
      link = drop(eta),
      class = drop(Matrix::Matrix(1 * (eta > 0))),
      drop(exp(eta) / (1 + exp(eta)))
    )
    return(out)
  }

  # Non-binomial
  if (type == "class") {
    stop("type = 'class' can only be used with family = 'binomial'", call. = FALSE)
  }
  drop(eta)
}

#' @method predict mbiglasso
#' @rdname predict.biglasso
#' @export

predict.mbiglasso <- function(
  object,
  X,
  row.idx = 1:nrow(X),
  type = c(
    "link",
    "response",
    "coefficients",
    "vars",
    "nvars"
  ),
  lambda = NULL,
  which = 1:length(object$lambda),
  k = 1,
  ...
) {
  type <- match.arg(type)
  beta <- coef.mbiglasso(object, lambda = lambda, which = which)[[k]]
  res <- predict_biglasso_common(beta, X, row.idx, type, strip_intercept = TRUE)
  if (res$done) return(res$value)
  res$eta
}

# Shared eta-computation/type-dispatch body behind predict.biglasso() and
# predict.mbiglasso(): both start from a beta matrix (intercept row
# included) and go coefficients-early-return -> optional intercept-stripping
# -> nvars/vars-early-return -> get_eta(). They differ only in whether the
# intercept row is stripped (`strip_intercept`, never done for family =
# "cox") and in what happens to eta afterwards -- the binomial link/class/
# response dispatch has no mgaussian equivalent, so it stays in
# predict.biglasso() itself rather than living here.
# Returns list(done = TRUE, value = ...) for the early-return types
# ("coefficients", "nvars", "vars"), or list(done = FALSE, eta = ...)
# otherwise.
predict_biglasso_common <- function(beta, X, row.idx, type, strip_intercept) {
  if (type == "coefficients") {
    return(list(done = TRUE, value = beta))
  }

  alpha <- NULL
  if (strip_intercept) {
    alpha <- beta[1, ]
    beta <- beta[-1, , drop = FALSE]
  }

  if (type == "nvars") {
    return(list(done = TRUE, value = Matrix::colSums(beta != 0, na.rm = TRUE)))
  }
  if (type == "vars") {
    return(list(done = TRUE, value = drop(apply(beta != 0, 2, FUN = which))))
  }

  if (!inherits(X, "big.matrix")) {
    stop("X must be a big.matrix object.", call. = FALSE)
  }

  beta_t <- as(beta, "TsparseMatrix")
  eta <- get_eta(X@address, as.integer(row.idx - 1), beta, beta_t@i, beta_t@j)
  if (!is.null(alpha)) eta <- sweep(eta, 2, alpha, "+")
  list(done = FALSE, eta = eta)
}

#' @method coef biglasso
#' @rdname predict.biglasso
#' @export

coef.biglasso <- function(object, lambda = NULL, which = 1:length(object$lambda), drop = TRUE, ...) {
  if (is.null(lambda)) {
    beta <- object$beta[, which, drop = FALSE]
  } else {
    if (max(lambda) > max(object$lambda) || min(lambda) < min(object$lambda)) {
      stop("Supplied lambda value(s) are outside the range of the model fit.", call. = FALSE)
    }
    ind <- stats::approx(object$lambda, seq(object$lambda), lambda)$y
    l <- floor(ind)
    r <- ceiling(ind)
    w <- ind %% 1
    beta <- (1 - w) * object$beta[, l, drop = FALSE] + w * object$beta[, r, drop = FALSE]
    colnames(beta) <- round(lambda, 4)
  }
  if (drop) {
    return(drop(beta))
  } else {
    return(beta)
  }
}

#' @method coef mbiglasso
#' @rdname predict.biglasso
#' @export

coef.mbiglasso <- function(object, lambda = NULL, which = 1:length(object$lambda), intercept = TRUE, ...) {
  nclass <- length(object$beta)
  beta <- vector("list", nclass)
  names(beta) <- names(object$beta)
  if (intercept) {
    col.idx <- 1:nrow(object$beta[[1]])
  } else {
    col.idx <- 2:nrow(object$beta[[1]])
  }
  if (is.null(lambda)) {
    for (class in 1:nclass) {
      beta[[class]] <- (object$beta[[class]])[col.idx, which, drop = FALSE]
    }
  } else {
    if (max(lambda) > max(object$lambda) || min(lambda) < min(object$lambda)) {
      stop("Supplied lambda value(s) are outside the range of the model fit.", call. = FALSE)
    }
    ind <- approx(object$lambda, seq(object$lambda), lambda)$y
    l <- floor(ind)
    r <- ceiling(ind)
    w <- ind %% 1
    for (class in 1:nclass) {
      beta_class <- (1 - w) *
        (object$beta[[class]])[col.idx, l, drop = FALSE] +
        w * (object$beta[[class]])[col.idx, r, drop = FALSE]
      colnames(beta_class) <- round(lambda, 4)
      beta[[class]] <- beta_class
    }
  }
  return(beta)
}
