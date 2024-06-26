% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/cv-biglasso.R
\name{cv.biglasso}
\alias{cv.biglasso}
\title{Cross-validation for biglasso}
\usage{
cv.biglasso(
  X,
  y,
  row.idx = 1:nrow(X),
  family = c("gaussian", "binomial", "cox", "mgaussian"),
  eval.metric = c("default", "MAPE", "auc", "class"),
  ncores = parallel::detectCores(),
  ...,
  nfolds = 5,
  seed,
  cv.ind,
  trace = FALSE,
  grouped = TRUE
)
}
\arguments{
\item{X}{The design matrix, without an intercept, as in \code{\link[=biglasso]{biglasso()}}.}

\item{y}{The response vector, as in \code{biglasso}.}

\item{row.idx}{The integer vector of row indices of \code{X} that used for
fitting the model. as in \code{biglasso}.}

\item{family}{Either \code{"gaussian"}, \code{"binomial"}, \code{"cox"} or
\code{"mgaussian"} depending on the response. \code{"cox"} and \code{"mgaussian"}
are not supported yet.}

\item{eval.metric}{The evaluation metric for the cross-validated error and
for choosing optimal \code{lambda}. "default" for linear regression is MSE
(mean squared error), for logistic regression is binomial deviance.
"MAPE", for linear regression only, is the Mean Absolute Percentage Error.
"auc", for binary classification, is the area under the receiver operating
characteristic curve (ROC).
"class", for binary classification, gives the misclassification error.}

\item{ncores}{The number of cores to use for parallel execution of the
cross-validation folds, run on a cluster created by the \code{parallel}
package. (This is also supplied to the \code{ncores} argument in
\code{\link[=biglasso]{biglasso()}}, which is the number of OpenMP threads, but only for
the first call of \code{\link[=biglasso]{biglasso()}} that is  run on the entire data. The
individual calls of \code{\link[=biglasso]{biglasso()}} for the CV folds are run without
the \code{ncores} argument.)}

\item{...}{Additional arguments to \code{biglasso}.}

\item{nfolds}{The number of cross-validation folds.  Default is 5.}

\item{seed}{The seed of the random number generator in order to obtain
reproducible results.}

\item{cv.ind}{Which fold each observation belongs to.  By default the
observations are randomly assigned by \code{cv.biglasso}.}

\item{trace}{If set to TRUE, cv.biglasso will inform the user of its
progress by announcing the beginning of each CV fold.  Default is FALSE.}

\item{grouped}{Whether to calculate CV standard error (\code{cvse}) over
CV folds (\code{TRUE}), or over all cross-validated predictions. Ignored
when \code{eval.metric} is 'auc'.}
}
\value{
An object with S3 class \code{"cv.biglasso"} which inherits from
class \code{"cv.ncvreg"}. The following variables are contained in the
class (adopted from \code{\link[ncvreg:cv.ncvreg]{ncvreg::cv.ncvreg()}}).
\item{cve}{The error for each value of \code{lambda}, averaged across the cross-validation folds.}
\item{cvse}{The estimated standard error associated with each value of for \code{cve}.}
\item{lambda}{The sequence of regularization parameter values along which the cross-validation error was calculated.}
\item{fit}{The fitted \code{biglasso} object for the whole data.}
\item{min}{The index of \code{lambda} corresponding to \code{lambda.min}.}
\item{lambda.min}{The value of \code{lambda} with the minimum cross-validation error.}
\item{lambda.1se}{The largest value of \code{lambda} for which the cross-validation error is at most one standard error larger than the minimum cross-validation error.}
\item{null.dev}{The deviance for the intercept-only model.}
\item{pe}{If \code{family="binomial"}, the cross-validation prediction error for each value of \code{lambda}.}
\item{cv.ind}{Same as above.}
}
\description{
Perform k-fold cross validation for penalized regression models over a grid
of values for the regularization parameter lambda.
}
\details{
The function calls \code{biglasso} \code{nfolds} times, each time leaving
out 1/\code{nfolds} of the data. The cross-validation error is based on the
residual sum of squares when \code{family="gaussian"} and the binomial
deviance when \code{family="binomial"}.

The S3 class object \code{cv.biglasso} inherits class \code{\link[ncvreg:cv.ncvreg]{ncvreg::cv.ncvreg()}}. So S3
functions such as \verb{"summary", "plot"} can be directly applied to the
\code{cv.biglasso} object.
}
\examples{
\dontrun{
## cv.biglasso
data(colon)
X <- colon$X
y <- colon$y
X.bm <- as.big.matrix(X)

## logistic regression
cvfit <- cv.biglasso(X.bm, y, family = 'binomial', seed = 1234, ncores = 2)
par(mfrow = c(2, 2))
plot(cvfit, type = 'all')
summary(cvfit)
}
}
\seealso{
\code{\link[=biglasso]{biglasso()}}, \code{\link[=plot.cv.biglasso]{plot.cv.biglasso()}}, \code{\link[=summary.cv.biglasso]{summary.cv.biglasso()}}, \code{\link[=setupX]{setupX()}}
}
\author{
Yaohui Zeng and Patrick Breheny
}
