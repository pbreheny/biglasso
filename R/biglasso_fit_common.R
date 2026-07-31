# Shared engine behind biglasso_fit() (single lambda) and biglasso_path()
# (a path of lambda values). The two public functions differ only in:
#   - the wording of the "lambda must be user-supplied" error (which must
#     stay in each public function: missing() can only see whether *that
#     function's own* argument was supplied, not one forwarded from a
#     caller)
#   - whether `lambda`/`dfmax` produce a single fit or a path (`path`)
# Every other validation/dispatch/output-assembly step is identical and
# lives here.
biglasso_fit_common <- function(X, y, r, init, xtx, penalty, lambda, alpha, gamma,
                                ncores, max.iter, eps, dfmax, penalty.factor, warn,
                                output.time, return.time, path) {
  if (missing(gamma)) gamma <- switch(penalty, SCAD = 3.7, 3)

  # check types
  if (!("big.matrix" %in% class(X)) || typeof(X) != "double") {
    stop("X must be a double type big.matrix.")
  }
  if (is.matrix(y)) y <- drop(y)

  if (any(is.na(y))) {
    stop(
      "Missing data (NA's) detected.  Take actions (e.g., removing cases, removing features, imputation) to eliminate missing data before fitting the model."
    )
  }

  if (!is.double(y)) {
    if (is.matrix(y)) {
      tmp <- try(storage.mode(y) <- "numeric", silent = TRUE)
    } else {
      tmp <- try(y <- as.numeric(y), silent = TRUE)
    }
    if (class(tmp)[1] == "try-error") stop("y must numeric or able to be coerced to numeric")
  }

  p <- ncol(X)
  if (length(penalty.factor) != p) stop("penalty.factor does not match up with X")
  storage.mode(penalty.factor) <- "double"

  n <- nrow(X)

  # check types for residuals and xtx
  if (!is.double(r)) r <- as.double(r)
  if (!is.double(xtx)) xtx <- as.double(xtx)

  ## fit model
  if (output.time) {
    cat("\nStart biglasso: ", format(Sys.time()), "\n")
  }

  if (path) {
    call_args <- list(
      X@address, y, r, init, xtx, penalty, lambda, length(lambda), alpha, gamma,
      eps, as.integer(dfmax), as.integer(max.iter), penalty.factor, as.integer(ncores)
    )
    routine <- "cdfit_gaussian_simple_path"
  } else {
    call_args <- list(
      X@address, y, r, init, xtx, penalty, lambda, alpha, gamma,
      eps, as.integer(dfmax), as.integer(max.iter), penalty.factor, as.integer(ncores)
    )
    routine <- "cdfit_gaussian_simple"
  }
  time <- system.time({
    res <- do.call(".Call", c(list(routine), call_args, list(PACKAGE = "biglasso")))
  })

  b <- res[[1]]
  loss <- res[[2]]
  iter <- res[[3]]
  resid <- res[[4]]

  if (path) {
    ind <- !is.na(iter)
    b <- b[, ind, drop = FALSE]
    loss <- loss[ind]
    iter <- iter[ind]
    lambda <- lambda[ind]
  }

  if (output.time) {
    cat("\nEnd biglasso: ", format(Sys.time()), "\n")
  }

  if (path) {
    if (warn && any(iter == max.iter)) {
      warning("Maximum number of iterations reached at ", sum(iter == max.iter), " lambda value(s)")
    }
  } else {
    # iter is NA rather than max.iter when the fit was cut short by dfmax
    if (warn && !is.na(iter) && iter == max.iter) warning("Maximum number of iterations reached")
  }

  ## Names
  varnames <- if (is.null(colnames(X))) paste("V", 1:p, sep = "") else colnames(X)
  if (path) rownames(b) <- varnames else names(b) <- varnames

  ## Output
  return.val <- list(
    beta = b,
    iter = iter,
    resid = resid,
    lambda = lambda,
    penalty = penalty,
    alpha = alpha,
    loss = loss,
    penalty.factor = penalty.factor,
    n = n,
    y = y
  )
  if (return.time) return.val$time <- as.numeric(time["elapsed"])

  structure(return.val, class = c("biglasso", "ncvreg"))
}
