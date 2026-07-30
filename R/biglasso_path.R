#' Direct interface to biglasso fitting, no preprocessing, path version
#' 
#' This function is intended for users who know exactly what they're doing and
#' want complete control over the fitting process. It
#' * does NOT add an intercept 
#' * does NOT standardize the design matrix
#' both of the above are critical steps in data analysis. However, a direct API
#' has been provided for use in situations where the lasso fitting process is
#' an internal component of a more complicated algorithm and standardization
#' must be handled externally.
#' 
#' `biglasso_path()` works identically to [biglasso_fit()] except it offers the
#' additional option of fitting models across a path of tuning parameter values.
#'  
#' Note:
#' * Hybrid safe-strong rules are turned off for `biglasso_fit()`, as these rely
#'   on standardization
#' * Currently, the function only works with linear regression
#'   (`family = 'gaussian'`).
#'  
#' @param X               The design matrix, without an intercept. It must be a
#'                        double type [bigmemory::big.matrix()] object. 
#' @param y               The response vector 
#' @param r               Residuals (length n vector) corresponding to `init`. 
#'                        WARNING: If you supply an incorrect value of `r`, the 
#'                        solution will be incorrect. 
#' @param init            Initial values for beta.  Default: zero (length p vector)
#' @param xtx             X scales: the jth element should equal `crossprod(X[,j])/n`.
#'                        In particular, if X is standardized, one should pass
#'                        `xtx = rep(1, p)`. WARNING: If you supply an incorrect value of
#'                        `xtx`, the solution will be incorrect. (length p vector)
#' @param penalty         String specifying which penalty to use. Default is 'lasso', 
#'                        Other options are 'SCAD' and 'MCP' (the latter are non-convex) 
#' @param lambda          A vector of numeric values the lasso tuning parameter. 
#' @param alpha           The elastic-net mixing parameter that controls the relative
#'                        contribution from the lasso (l1) and the ridge (l2) penalty. 
#'                        The penalty is defined as:
#'                        \deqn{ \alpha||\beta||_1 + (1-\alpha)/2||\beta||_2^2.}
#'                        `alpha=1` is the lasso penalty, `alpha=0` the ridge penalty,
#'                        `alpha` in between 0 and 1 is the elastic-net ("enet") penalty.
#' @param gamma           Tuning parameter value for nonconvex penalty. Defaults are
#'                        3.7 for `penalty = 'SCAD'` and 3 for `penalty = 'MCP'`
#' @param ncores          The number of OpenMP threads used for parallel computing.
#' @param max.iter        Maximum number of iterations.  Default is 1000.
#' @param eps             Convergence threshold for inner coordinate descent. The
#'                        algorithm iterates until the maximum change in the objective 
#'                        after any coefficient update is less than `eps` times 
#'                        the null deviance. Default value is `1e-7`.
#' @param dfmax           Upper bound for the number of nonzero coefficients. Default is
#'                        no upper bound.  However, for large data sets, 
#'                        computational burden may be heavy for models with a large 
#'                        number of nonzero coefficients.
#' @param penalty.factor  A multiplicative factor for the penalty applied to
#'                        each coefficient. If supplied, `penalty.factor` must be a numeric
#'                        vector of length equal to the number of columns of `X`.  
#' @param warn            Return warning messages for failures to converge and model
#'                        saturation?  Default is TRUE.
#' @param output.time     Whether to print out the start and end time of the model
#'                        fitting. Default is FALSE.
#' @param return.time     Whether to return the computing time of the model
#'                        fitting. Default is TRUE.
#'                        
#' @returns An object with S3 class `"biglasso"` with following variables.
#' \item{beta}{A sparse matrix where rows are estimates a given coefficient across all values of lambda} 
#' \item{iter}{A vector of length `nlambda` containing the number of 
#' iterations until convergence} 
#' \item{resid}{Vector of residuals calculated from estimated coefficients.}
#' \item{lambda}{The sequence of regularization parameter values in the path.}
#' \item{alpha}{Same as in `biglasso()`} 
#' \item{loss}{A vector containing either the residual sum of squares of the fitted model at each value of lambda.}
#' \item{penalty.factor}{Same as in `biglasso()`.}
#' \item{n}{The number of observations used in the model fitting.}
#' \item{y}{The response vector used in the model fitting.}
#' 
#' @author Tabitha Peter and Patrick Breheny 
#'
#' @examples
#' data(Prostate)
#' X <- cbind(1, Prostate$X)
#' xtx <- apply(X, 2, crossprod)/nrow(X)
#' y <- Prostate$y
#' X.bm <- as.big.matrix(X)
#' init <- rep(0, ncol(X))
#' fit <- biglasso_path(X = X.bm, y = y, r = y, init = init, xtx = xtx,
#'   lambda = c(0.5, 0.1, 0.05, 0.01, 0.001), 
#'   penalty.factor=c(0, rep(1, ncol(X)-1)), max.iter=2000)
#' fit$beta
#'   
#' fit <- biglasso_path(X = X.bm, y = y, r = y, init = init, xtx = xtx,
#'   lambda = c(0.5, 0.1, 0.05, 0.01, 0.001), penalty='MCP',
#'   penalty.factor=c(0, rep(1, ncol(X)-1)), max.iter = 2000)
#' fit$beta
#' @export biglasso_path

biglasso_path <- function(X,
                          y,
                          r,
                          init = rep(0, ncol(X)),
                          xtx,
                          penalty = "lasso",
                          lambda,
                          alpha = 1,
                          gamma,
                          ncores = 1,
                          max.iter = 1000,
                          eps = 1e-5,
                          dfmax = ncol(X)+1,
                          penalty.factor = rep(1, ncol(X)),
                          warn = TRUE,
                          output.time = FALSE,
                          return.time = TRUE) {
  if (missing(lambda)) {
    stop("For biglasso_path, a vector of lambda values must be user-supplied")
  }
  biglasso_fit_common(
    X = X, y = y, r = r, init = init, xtx = xtx, penalty = penalty, lambda = lambda,
    alpha = alpha, gamma = gamma, ncores = ncores, max.iter = max.iter, eps = eps,
    dfmax = dfmax, penalty.factor = penalty.factor, warn = warn,
    output.time = output.time, return.time = return.time, path = TRUE
  )
}
