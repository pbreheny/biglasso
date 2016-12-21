\name{biglasso}

\alias{biglasso}

\title{Fit lasso penalized regression path for big data}

\description{
Extend lasso model fitting to big data that cannot be loaded into memory. Fit 
solution paths for linear or logistic regression models penalized by lasso, ridge, or elastic-net over a grid of values for the regularization parameter lambda.
}

\usage{
biglasso(X, y, row.idx = 1:nrow(X), penalty = c("lasso", "ridge", "enet"),
         family = c("gaussian","binomial"), alg.logistic = c("Newton", "MM"),
         screen = c("SSR", "SEDPP", "SSR-BEDPP", "SSR-Slores", 
                    "SSR-Dome", "SEDPP-No-Active", "None"),
         safe.thresh = 0, ncores = 1, alpha = 1, 
         lambda.min = ifelse(nrow(X) > ncol(X),.001,.05), 
         nlambda = 100, lambda.log.scale = TRUE, lambda, eps = 1e-7, 
         max.iter = 1000, dfmax = ncol(X)+1, penalty.factor = rep(1, ncol(X)), 
         warn = TRUE, output.time = FALSE, verbose = FALSE)
}

\arguments{
  \item{X}{The design matrix, without an intercept. It must be a \code{\link[bigmemory]{big.matrix}} object. The function standardizes the data and includes an intercept internally by default during the model fitting.}
  \item{y}{The response vector.}
  \item{row.idx}{The integer vector of row indices of \code{X} that used for fitting 
  the model. \code{1:nrow(X)} by default.}
  \item{penalty}{The penalty to be applied to the model. Either "lasso" (the default), 
  "ridge", or "enet" (elastic net).}
  \item{family}{Either "gaussian" or "binomial", depending on the response.}
  \item{alg.logistic}{The algorithm used in logistic regression. If "Newton" then the exact hessian is used (default); if "MM" then a majorization-minimization algorithm is used to set an upper-bound on the hessian matrix. This can be faster, particularly in data-larger-than-RAM case.}
  \item{screen}{The feature screening rule used at each \code{lambda} that discards features to speed up computation: "SSR" (default) is the sequential strong rule; "SEDPP" is the (sequential) EDPP rule. "SSR-BEDPP", "SSR-Dome", and "SSR-Slores" are our newly proposed screening rules which combine the strong rule with a safe rule (BEDPP, Dome test, or Slores rule). Among the three, the first two are for lasso-penalized linear regression, and the last one is for lasso-penalized logistic regression. Besides, "SEDPP-No-Active" is the (sequential) EDPP rule without active set cycling in the algorithm, which is for research purpose only. "None" is to not apply a rule. \strong{Note that for linear regression with elastic net penalty, both "SSR" and "SSR-BEDPP" are applicable since version 1.3-0.} Only "SSR" is applicable to elastic-net-penalized logistic regression.
  }
  \item{safe.thresh}{the threshold value between 0 and 1 that controls when to stop safe test in the "SSR-Dome" and "SSR-BEDPP" rules. For example, 0.01 means to stop Dome test at next lambda iteration if the number of features rejected by safe test at current lambda iteration is not larger than 1\% of the total number of features. So 1 means to always turn off safe test, whereas 0 (default) means to turn off safe test if the number of features rejected by safe test is 0 at current lambda.}
  \item{ncores}{The number of OpenMP threads used for parallel computing.}
  \item{alpha}{The elastic-net mixing parameter that controls the relative contribution from the lasso (l1) and the ridge (l2) penalty. The penalty is defined as \deqn{ \alpha||\beta||_1  + (1-\alpha)/2||\beta||_2^2.} \code{alpha=1} is the lasso penalty, \code{alpha=0} the ridge penalty, \code{alpha} in between 0 and 1 is the elastic-net ("enet") penalty. }
  \item{lambda.min}{The smallest value for lambda, as a fraction of
    lambda.max.  Default is .001 if the number of observations is larger
    than the number of covariates and .05 otherwise.}
  \item{nlambda}{The number of lambda values.  Default is 100.}
  \item{lambda.log.scale}{Whether compute the grid values of lambda on log scale (default) or linear scale.}
  \item{lambda}{A user-specified sequence of lambda values.  By default,
    a sequence of values of length \code{nlambda} is computed, equally
    spaced on the log scale.}
  \item{eps}{Convergence threshold for inner coordinate descent.  The algorithm iterates until the maximum change in the objective after any coefficient update is less than \code{eps} times the null deviance. Default value is \code{1e-7}.}
  \item{max.iter}{Maximum number of iterations.  Default is 1000.}
  \item{dfmax}{Upper bound for the number of nonzero coefficients.
    Default is no upper bound.  However, for large data sets,
    computational burden may be heavy for models with a large number of
    nonzero coefficients.}
  \item{penalty.factor}{A multiplicative factor for the penalty applied
    to each coefficient.  If supplied, \code{penalty.factor} must be a
    numeric vector of length equal to the number of columns of
    \code{X}.  The purpose of \code{penalty.factor} is to apply
    differential penalization if some coefficients are thought to be
    more likely than others to be in the model. Current package doesn't allow
    unpenalized coefficients. That is\code{penalty.factor} cannot be 0. }
  \item{warn}{Return warning messages for failures to converge and model
    saturation?  Default is TRUE.}
  \item{output.time}{Whether to print out the start and end time of the model fitting.}
  \item{verbose}{Whether to output the timing of each lambda iteration. Default is FALSE.}
}

\details{
The objective function for linear regression (\code{family = "gaussian"}) is \deqn{\frac{1}{2n}\textrm{RSS} + \textrm{penalty},} for logistic regression (\code{family = "binomial"}) it is
  \deqn{-\frac{1}{n} loglike + \textrm{penalty}.} 
  
Several advanced feature screening rules are implemented. For lasso-penalized linear regression, all the options of \code{screen} are applicable. Our proposal rule - "SSR-BEDPP" - achieves highest speedup so it's the recommended one, especially for ultrahigh-dimensional large-scale data sets. For logistic regression and/or the elastic net penalty, only "SSR" is applicable for now. More efficient rules are under development.
}

\value{
  An object with S3 class \code{"biglasso"} with following variables.
  \item{beta}{The fitted matrix of coefficients, store in sparse matrix representation. The number of rows is equal to the number of coefficients, whereas the number of columns is equal to \code{nlambda}.}
  \item{iter}{A vector of length \code{nlambda} containing the number of iterations until convergence at each value of \code{lambda}.}
  \item{lambda}{The sequence of regularization parameter values in the path.}
  \item{penalty}{Same as above.}
  \item{family}{Same as above.}
  \item{alpha}{Same as above.}
  \item{loss}{A vector containing either the residual sum of squares (\code{for "gaussian"}) or negative log-likelihood (for \code{"binomial"}) of the fitted model at each value of \code{lambda}.}
  \item{penalty.factor}{Same as above.}
  \item{n}{The number of observations used in the model fitting. It's equal to \code{length(row.idx)}.}
  \item{center}{The sample mean vector of the variables, i.e., column mean of the sub-matrix of \code{X} used for model fitting.}
  \item{scale}{The sample standard deviation of the variables, i.e., column standard deviation of the sub-matrix of \code{X} used for model fitting.}
  \item{y}{The response vector used in the model fitting. Depending on \code{row.idx}, it could be a subset of the raw input of the response vector y.}
  \item{screen}{Same as above.}
  \item{col.idx}{The indices of features that have 'scale' value greater than 1e-6. Features with 'scale' less than 1e-6 are removed from model fitting.}
  \item{rejections}{The number of features rejected at each value of \code{lambda}.}
  \item{safe_rejections}{The number of features rejected by safe rules at each value of \code{lambda}. Only for "SSR-Dome" and "SSR-BEDPP" cases.}
}

\author{
  Yaohui Zeng and Patrick Breheny
  
  Maintainer: Yaohui Zeng <yaohui-zeng@uiowa.edu>
}

\seealso{\code{\link{biglasso-package}}, \code{\link{setupX}}, \code{\link{cv.biglasso}}, \code{\link{plot.biglasso}}, \code{\link[ncvreg]{ncvreg}}}
\examples{
## Linear regression
data(colon)
X <- colon$X
y <- colon$y
X.bm <- as.big.matrix(X)
# lasso, default
par(mfrow=c(1,2))
fit.lasso <- biglasso(X.bm, y, family = 'gaussian')
plot(fit.lasso, log.l = TRUE, main = 'lasso')
# elastic net
fit.enet <- biglasso(X.bm, y, penalty = 'enet', alpha = 0.5, family = 'gaussian')
plot(fit.enet, log.l = TRUE, main = 'elastic net, alpha = 0.5')

## Logistic regression
data(colon)
X <- colon$X
y <- colon$y
X.bm <- as.big.matrix(X)
# lasso, default
par(mfrow = c(1, 2))
fit.bin.lasso <- biglasso(X.bm, y, penalty = 'lasso', family = "binomial")
plot(fit.bin.lasso, log.l = TRUE, main = 'lasso')
# elastic net
fit.bin.enet <- biglasso(X.bm, y, penalty = 'enet', alpha = 0.5, family = "binomial")
plot(fit.bin.enet, log.l = TRUE, main = 'elastic net, alpha = 0.5')
}
