#' Simplified call to biglasso: a gaussian model fit with no 'bells and whistles' (e.g., no SSR)
#' 
#' NOTE: this function is designed for users who have a strong understanding of 
#' statistics and know exactly what they are doing. This is a simplification of  
#' the main `biglasso()` function with more flexible settings. 
#' 
#' Of note, this function:
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
#' @param X               The design matrix, without an intercept. It must be a
#'                        double type \code{\link[bigmemory]{big.matrix}} object. 
#' @param y               The response vector 
#' @param r               Residuals (length n vector) corresponding to `init`. 
#'                        WARNING: If you supply an incorrect value of `r`, the 
#'                        solution will be incorrect. 
#' @param init            Initial values for beta.  Default: zero (length p vector)
#' @param xtx             X scales: the jth element should equal `crossprod(X[,j])/n`.
#'                        In particular, if X is standardized, one should pass
#'                        `xtx = rep(1, p)`.  WARNING: If you supply an incorrect value of
#'                        `xtx`, the solution will be incorrect. (length p vector)
#' @param lambda          A single value for the lasso tuning parameter. 
#' @param alpha           The elastic-net mixing parameter that controls the relative
#'                        contribution from the lasso (l1) and the ridge (l2) penalty. 
#'                        The penalty is defined as:
#'                        \deqn{ \alpha||\beta||_1 + (1-\alpha)/2||\beta||_2^2.}
#'                        \code{alpha=1} is the lasso penalty, \code{alpha=0} the ridge penalty,
#'                        \code{alpha} in between 0 and 1 is the elastic-net ("enet") penalty.
#' @param ncores          The number of OpenMP threads used for parallel computing.
#' @param max.iter        Maximum number of iterations.  Default is 1000.
#' @param eps             Convergence threshold for inner coordinate descent. The
#'                        algorithm iterates until the maximum change in the objective 
#'                        after any coefficient update is less than \code{eps} times 
#'                        the null deviance. Default value is \code{1e-7}.
#' @param dfmax           Upper bound for the number of nonzero coefficients. Default is
#'                        no upper bound.  However, for large data sets, 
#'                        computational burden may be heavy for models with a large 
#'                        number of nonzero coefficients.
#' @param penalty.factor  A multiplicative factor for the penalty applied to
#'                        each coefficient. If supplied, \code{penalty.factor} must be a numeric
#'                        vector of length equal to the number of columns of \code{X}.  
#' @param warn            Return warning messages for failures to converge and model
#'                        saturation?  Default is TRUE.
#' @param output.time     Whether to print out the start and end time of the model
#'                        fitting. Default is FALSE.
#' @param return.time     Whether to return the computing time of the model
#'                        fitting. Default is TRUE.
#'                        
#' @return An object with S3 class \code{"biglasso"} with following variables.
#' \item{beta}{The vector of estimated coefficients} 
#' \item{iter}{A vector of length \code{nlambda} containing the number of 
#' iterations until convergence} 
#' \item{resid}{Vector of residuals calculated from estimated coefficients.}
#' \item{lambda}{The sequence of regularization parameter values in the path.}
#' \item{alpha}{Same as in `biglasso()`} 
#' \item{loss}{A vector containing either the residual sum of squares of the fitted model at each value of lambda.}
#' \item{penalty.factor}{Same as in `biglasso()`.}
#' \item{n}{The number of observations used in the model fitting.}
#' \item{y}{The response vector used in the model fitting.}
#' @author Yaohui Zeng, Chuyi Wang, Tabitha Peter, and Patrick Breheny 
#'
#' @examples
#' 
#' data(Prostate)
#' X <- cbind(1, Prostate$X) |> ncvreg::std() # standardizing -> xtx is all 1s
#' y <- Prostate$y
#' X.bm <- as.big.matrix(X)
#' init <- rep(0, ncol(X)) # using cold starts - will need more iterations
#' r <- y - X%*%init
#' fit_flex <- biglasso_fit(X = X.bm, y = y, r = r, init = init,
#'  xtx = rep(1, ncol(X)),lambda = 0.1, penalty.factor=c(0, rep(1, ncol(X)-1)),
#'   max.iter = 10000)
#' @export biglasso_fit
biglasso_fit <- function(X,
                         y,
                         r, 
                         init=rep(0, ncol(X)),
                         xtx, 
                         lambda,
                         alpha = 1, 
                         ncores = 1,
                         max.iter = 1000, 
                         eps=1e-5,
                         dfmax = ncol(X)+1,
                         penalty.factor = rep(1, ncol(X)),
                         warn = TRUE,
                         output.time = FALSE,
                         return.time = TRUE) {

  # set defaults
  penalty <- "lasso"
  alpha <- 1
  
  # check types
  if (!("big.matrix" %in% class(X)) || typeof(X) != "double") stop("X must be a double type big.matrix.")
  # subset of the response vector
  if (is.matrix(y)) y <- drop(y)
  else y <- y
  
  if (any(is.na(y))) stop("Missing data (NA's) detected.  Take actions (e.g., removing cases, removing features, imputation) to eliminate missing data before fitting the model.")
  
  if (!is.double(y)) {
    if (is.matrix(y)) tmp <- try(storage.mode(y) <- "numeric", silent=TRUE)
    else tmp <- try(y <- as.numeric(y), silent=TRUE)
    if (class(tmp)[1] == "try-error") stop("y must numeric or able to be coerced to numeric")
  }
  
  p <- ncol(X)
  if (length(penalty.factor) != p) stop("penalty.factor does not match up with X")
  
  storage.mode(penalty.factor) <- "double"
  
  n <- nrow(X) 
  if (missing(lambda)) {
    stop("For biglasso_fit, a single lambda value must be user-supplied")
  }
  
  # check types for residuals and xtx
  if (!is.double(r)) r <- as.double(r)
  if (!is.double(xtx)) xtx <- as.double(xtx)
  
  ## fit model
  if (output.time) {
    cat("\nStart biglasso: ", format(Sys.time()), '\n')
  }
  
  Sys.setenv(R_C_BOUNDS_CHECK = "yes")

  time <- system.time(
    res <- .Call("cdfit_gaussian_simple",
                 X@address,
                 y,
                 r,
                 init, 
                 xtx,
                 lambda,
                 alpha,
                 eps,
                 as.integer(max.iter),
                 penalty.factor,
                 as.integer(ncores),
                 PACKAGE = 'biglasso')
    
   
  )

  b <- res[[1]]
  loss <- res[[2]]
  iter <- res[[3]]
  resid <- res[[4]] # TODO: think about whether I need to add this in 
  
  if (output.time) {
    cat("\nEnd biglasso: ", format(Sys.time()), '\n')
  }
 
  if (warn & (iter==max.iter)) {
    warning("Maximum number of iterations reached")
  }
  
  
  ## Names
  names(b) <- if (is.null(colnames(X))) paste("V", 1:p, sep="") else colnames(X)
  
  ## Output
  return.val <- list(
    beta = b,
    iter = iter,
    resid = resid,
    lambda = lambda,
    # TODO: will need to add these later
    # penalty = penalty,
    # family = family,
    alpha = alpha,
    loss = loss,
    penalty.factor = penalty.factor,
    n = n,
    y = y
  )
  
  if (return.time) return.val$time <- as.numeric(time['elapsed'])
  
  val <- structure(return.val, class = c("biglasso", 'ncvreg'))
}

