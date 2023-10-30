#' COPY of biglasso, adapted to have more flexible settings 
#' 
#' NOTE: this function is designed for users who have a strong understanding of 
#' statistics and know exactly what they are doing. This is a copy of the main 
#' `biglasso()` function with more flexible settings. Of note, this function:
#' 
#'  * does NOT add an intercept 
#'  * does NOT standardize the design matrix
#'  * does NOT set up a path for lambda (the lasso tuning parameter)
#'  
#'  all of the above are among the best practices for data analysis. This function 
#'  is made for use in situations where these steps have already been addressed prior 
#'  to model fitting.
#'  
#'  In other words, `biglasso_fit()` is to `biglasso()` what `ncvreg::ncvfit()`
#'  is to `ncvreg::ncvreg()`.
#'  
#'  For now, this function only works with linear regression (`family = 'gaussian'`)
#'  
#' @param X The design matrix, without an intercept. It must be a
#' double type \code{\link[bigmemory]{big.matrix}} object. 
#' @param y The response vector 
#' @param row.idx The integer vector of row indices of \code{X} that used for
#' fitting the model. \code{1:nrow(X)} by default.
#' @param ncores The number of OpenMP threads used for parallel computing.
#' @param lambda A single value for the lasso tuning parameter. 
#' @param alpha The elastic-net mixing parameter that controls the relative
#' contribution from the lasso (l1) and the ridge (l2) penalty. The penalty is
#' defined as \deqn{ \alpha||\beta||_1 + (1-\alpha)/2||\beta||_2^2.}
#' \code{alpha=1} is the lasso penalty, \code{alpha=0} the ridge penalty,
#' \code{alpha} in between 0 and 1 is the elastic-net ("enet") penalty.
#' @param max.iter Maximum number of iterations.  Default is 1000.
#' @param eps Convergence threshold for inner coordinate descent.  The
#' algorithm iterates until the maximum change in the objective after any
#' coefficient update is less than \code{eps} times the null deviance. Default
#' value is \code{1e-7}.
#' @param dfmax Upper bound for the number of nonzero coefficients.  Default is
#' no upper bound.  However, for large data sets, computational burden may be
#' heavy for models with a large number of nonzero coefficients.
#' @param penalty.factor A multiplicative factor for the penalty applied to
#' each coefficient. If supplied, \code{penalty.factor} must be a numeric
#' vector of length equal to the number of columns of \code{X}.  
#' @param warn Return warning messages for failures to converge and model
#' saturation?  Default is TRUE.
#' @param output.time Whether to print out the start and end time of the model
#' fitting. Default is FALSE.
#' @param return.time Whether to return the computing time of the model
#' fitting. Default is TRUE.
#' @param verbose Whether to output the timing of each lambda iteration.
#' Default is FALSE.
#' @return An object with S3 class \code{"biglasso"} with following variables.
#' \item{beta}{The fitted matrix of coefficients, store in sparse matrix
#' representation. The number of rows is equal to the number of coefficients,
#' whereas the number of columns is equal to \code{nlambda}. For \code{"mgaussian"}
#' family with m responses, it is a list of m such matrices.} 
#' \item{iter}{A vector of length \code{nlambda} containing the number of 
#' iterations until convergence at each value of \code{lambda}.} 
#' \item{lambda}{The sequence of regularization parameter values in the path.}
#' \item{penalty}{Same as above.}
#' \item{family}{Same as above.}
#' \item{alpha}{Same as above.} 
#' \item{loss}{A vector containing either the residual sum of squares of the fitted model at each value of \code{lambda}.}
#' \item{penalty.factor}{Same as above.}
#' \item{n}{The number of observations used in the model fitting. It's equal to
#' \code{length(row.idx)}.} 
#' \item{y}{The response vector used in the model fitting. Depending on
#' \code{row.idx}, it could be a subset of the raw input of the response vector y.}
#' \item{screen}{Same as above.} 
#' \item{col.idx}{The indices of features that have 'scale' value greater than
#' 1e-6. Features with 'scale' less than 1e-6 are removed from model fitting.} 
#' \item{rejections}{The number of features rejected at each value of \code{lambda}.}
#' \item{safe_rejections}{The number of features rejected by safe rules at each
#' value of \code{lambda}.}
#' @author Yaohui Zeng, Chuyi Wang, Tabitha Peter, and Patrick Breheny 
#'
#' @examples
#' 
#' data(colon)
#' X <- colon$X
#' y <- colon$y
#' X.bm <- as.big.matrix(X)
#' fit_flex <- biglasso_fit(X.bm, y, lambda = 0.1)
#' plot(fit_flex, log.l = TRUE, main = 'lasso')
#'
#' @export biglasso_fit
biglasso_fit <- function(X, y,
                         row.idx = 1:nrow(X),
                         ncores = 1,
                         lambda,
                         alpha = 1, 
                         max.iter = 1000, 
                         eps=1e-5,
                         dfmax = ncol(X)+1,
                         penalty.factor = rep(1, ncol(X)),
                         warn = TRUE,
                         output.time = FALSE,
                         return.time = TRUE,
                         verbose = FALSE) {
  
  # set defaults
  penalty <- "lasso"
  alpha <- 1
  
  # check types
  if (!("big.matrix" %in% class(X)) || typeof(X) != "double") stop("X must be a double type big.matrix.")
  # subset of the response vector
  if (is.matrix(y)) y <- y[row.idx,]
  else y <- y[row.idx]
  
  if (any(is.na(y))) stop("Missing data (NA's) detected.  Take actions (e.g., removing cases, removing features, imputation) to eliminate missing data before fitting the model.")
  
  if (!is.double(y)) {
    if (is.matrix(y)) tmp <- try(storage.mode(y) <- "numeric", silent=TRUE)
    else tmp <- try(y <- as.numeric(y), silent=TRUE)
    if (class(tmp)[1] == "try-error") stop("y must numeric or able to be coerced to numeric")
  }
  
  p <- ncol(X)
  if (length(penalty.factor) != p) stop("penalty.factor does not match up with X")
  
  storage.mode(penalty.factor) <- "double"
  
  n <- length(row.idx) ## subset of X. idx: indices of rows.
  if (missing(lambda)) {
    stop("For biglasso_fit, a single lambda value must be user-supplied")
  }
  
  ## fit model
  if (output.time) {
    cat("\nStart biglasso: ", format(Sys.time()), '\n')
  }
  
  time <- system.time(
    res <- .Call("cdfit_gaussian_simple",
                 X@address,
                 y,
                 as.integer(row.idx-1),
                 lambda,
                 alpha,
                 eps,
                 as.integer(max.iter),
                 penalty.factor,
                 as.integer(ncores),
                 verbose,
                 PACKAGE = 'biglasso')
    
  )
  
  
  b <- Matrix(res[[1]], sparse = T)
  res_lam <- res[[2]]
  loss <- res[[3]]
  iter <- res[[4]]
  resid <- res[[5]]
  rejections <- res[[6]]
  safe_rejections <- res[[7]]
  col.idx <- res[[8]]
  
  
  if (output.time) {
    cat("\nEnd biglasso: ", format(Sys.time()), '\n')
  }
  
  ## Eliminate saturated lambda values, if any
  ind <- !is.na(iter)
  if(!is.list(b)) b <- b[, ind, drop=FALSE]
  iter <- iter[ind]
  lambda <- lambda[ind]
  loss <- loss[ind]
  
  if (warn & any(iter==max.iter)) warning("Algorithm failed to converge for some values of lambda")
  
  ## Names
  varnames <- if (is.null(colnames(X))) paste("V", 1:p, sep="") else colnames(X)
  dimnames(b) <- list(varnames, round(lambda, digits = 4))
  
  ## Output
  return.val <- list(
    beta = b,
    iter = iter,
    lambda = lambda,
    penalty = penalty,
    family = family,
    alpha = alpha,
    loss = loss,
    penalty.factor = penalty.factor,
    n = n,
    y = y,
    screen = screen,
    col.idx = col.idx,
    # safe_rejections = safe_rejections,
    rejections = rejections
    
  )
  
  if (return.time) return.val$time <- as.numeric(time['elapsed'])
  
  val <- structure(return.val, class = c("biglasso", 'ncvreg'))
}

