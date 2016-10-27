\name{biglasso-package}

\alias{biglasso-package}

\docType{package}

\title{Big Lasso: Extending Lasso Model Fitting to Big Data in R}

\description{
Extend lasso and elastic-net model fitting for ultrahigh-dimensional, multi-gigabyte data sets that cannot be loaded into memory. This package utilizes memory-mapped files to store the massive data on the disk and only read those into memory whenever necessary during model fitting. Compared to existing lasso-fitting packages such as \href{http://cran.r-project.org/package=glmnet}{glmnet} and \href{http://cran.r-project.org/package=ncvreg}{ncvreg}, it preserves equivalently fast computation speed but is much more memory-efficient, thus allowing for very powerful big data analysis even with only a single laptop.
}

\details{
  \tabular{ll}{
    Package: \tab biglasso\cr
    Type: \tab Package\cr
    Version: \tab 1.2-2\cr
    Date: \tab 2016-10-27\cr
    License: \tab GPL-2\cr}
 
Penalized regression models, in particular the lasso, have been extensively applied to analyzing high-dimensional data sets. However, due to the memory limit, existing R packages are not capable of fitting lasso models for ultrahigh-dimensional, multi-gigabyte data sets that have been increasingly seen in many areas such as genetics, biomedical imaging, genome sequencing and high-frequency finance.

Built upon existing C++ library "bigmemory", this package tackles the big data challenge by storing the massive data on the disk and only read those into memory whenever necessary during model fitting, thus can easily handle cases where the data sets are larger than avaiable RAM. Current benchmarking experiments demonstrate that the model fitting takes less than 3 minutes for a 10 GB data (1000 observations, 1 million variables) on an ordinary computer.

The input design matrix X must be a \code{\link[bigmemory]{big.matrix}} object. This can be created by using the function \code{\link[bigmemory]{as.big.matrix}} in the R package \href{http://cran.r-project.org/package=bigmemory}{bigmemory}. If the data (design matrix) is very large (e.g. 10 GB) and stored in an external file, which is often the case for big data, X can be created by the function \code{\link{setupX}}. \strong{In this case, there are several restrictions about the data file:} 

\enumerate{
\item the data file must be a well-formated ASCII-file, with each row corresponding to an observation and each column a variable;
  \item the data file must contain only one single type. Current version only supports \code{double} type;
  \item the data file must contain only numeric variables. If there are categorical variables, the user needs to create dummy variables for each categorical varable (by adding additional columns).
}

Future versions will try to address these restrictions.
}

\note{
Denote the number of observations and variables be, respectively, \code{n} and \code{p}. It's worth noting that the package is more suitable for wide data (ultrahigh-dimensional, \code{p >> n}) as compared to long data (\code{n >> p}). This is because the model fitting algorithm takes advantage of sparsity assumption of high-dimensional data.

To just give the user some ideas, below are some benchmarking results of the time used for fitting a single lasso model with different sizes of the design matrix. More rigorous benchmarking results will be reported later. 

\itemize{
\item For wide data case (\code{p >> n}), \code{n = 1,000}:
\tabular{ccccc}{
\code{p} \tab 1,000 \tab 10,000 \tab 100,000 \tab 1,000,000 \cr
Size of \code{X} \tab 9.5 MB \tab 95 MB \tab 950 MB \tab 9.5 GB \cr
Elapsed time (s) \tab 0.22 \tab 1.18 \tab 12.34 \tab 145.40 \cr
}
\item For long data case (\code{n >> p}), \code{p = 1,000}:
\tabular{ccccc}{
\code{n} \tab 1,000 \tab 10,000 \tab 100,000 \tab 1,000,000 \cr
Size of \code{X} \tab 9.5 MB \tab 95 MB \tab 950 MB \tab 9.5 GB \cr
Elapsed time (s) \tab 2.50 \tab 11.43 \tab 83.69 \tab 1090.62 \cr
}
}

}


\author{
  Yaohui Zeng and Patrick Breheny
  
  Maintainer: Yaohui Zeng <yaohui-zeng@uiowa.edu>
}

\keyword{package}

\examples{
\dontrun{
## Example of reading data from external big data file, fit lasso model, run
## cross validation

# simulated design matrix, 1000 observations, 500,000 variables, ~ 5GB
# there are 10 true variables with non-zero coefficient 2.
xfname <- 'x_e3_5e5.txt' 
yfname <- 'y_e3_5e5.txt' # response vector
time <- system.time(
  X <- setupX(xfname, sep = '\t')
)
print(time) # ~ 8 minutes; this is just one time operation
dim(X)
y <- as.matrix(read.table(yfname, header = F))
time.fit <- system.time(
  fit <- biglasso(X, y, family = 'gaussian')
)
print(time.fit) # ~ 1 minute for fitting a lasso model

# cross validation in parallel
seed <- 1234
time.cvfit <- system.time(
  cvfit <- cv.biglasso(X, y, family = 'gaussian', seed = seed, ncores = 5)
)
print(time.cvfit) # ~ 4 minutes for 10-fold cross validation
plot(cvfit)
summary(cvfit)

# the big.matrix can be retrived by its descriptor file
rm(list = ls())
# the descriptor file was created after calling setupX(), and stored on the disk
xdesc <- 'x_e3_5e5.desc' 
X <- attach.big.matrix(xdesc)
dim(X)
}
}
