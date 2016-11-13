\name{biglasso-package}

\alias{biglasso-package}

\docType{package}

\title{Big Lasso: Extending Lasso Model Fitting to Big Data in R}

\description{
Extend lasso and elastic-net model fitting for ultrahigh-dimensional, multi-gigabyte data sets that cannot be loaded into available RAM. This package utilizes memory-mapped files to store the massive data on the disk and only read those into memory whenever necessary during model fitting. Moreover, some advanced feature screening rules are proposed and implemented to accelerate the model fitting. As a result, this package is much more memory- and computation-efficient as compared to existing lasso-fitting packages such as \href{https://CRAN.R-project.org/package=glmnet}{glmnet} and \href{https://CRAN.R-project.org/package=ncvreg}{ncvreg}, thus allowing for very powerful big data analysis even with only an ordinary laptop.
}

\details{
  \tabular{ll}{
    Package: \tab biglasso\cr
    Type: \tab Package\cr
    Version: \tab 1.2-3\cr
    Date: \tab 2016-11-13\cr
    License: \tab GPL-2\cr}
 
Penalized regression models, in particular the lasso, have been extensively applied to analyzing high-dimensional data sets. However, due to the memory limit, existing R packages are not capable of fitting lasso models for ultrahigh-dimensional, multi-gigabyte data sets which have been increasingly seen in many areas such as genetics, biomedical imaging, genome sequencing and high-frequency finance. 

This package aims to fill the gap by extending lasso model fitting to Big Data in R. Version >= 1.2-3 represents a major redesign where the source code is converted into C++ (previously in C), and new feature screening rules, as well as OpenMP parallel computing, are implemented. Some key features of \code{biglasso} are summarized as below:
\enumerate{
\item It utilizes memory-mapping files to store the massive data on the disk and only reads those into memory whenever necessary during model fitting, thus can easily handle data-larger-than-RAM cases.
\item It builds upon the pathwise coordinate descent algorithm combined with "warm start", "active set cycling", and feature screening strategies, which has been proved to be one of the most powerful lasso solvers.
\item It incorporates some efficient feature screening rules, such as the sequential strong rule (SSR), the sequential EDPP rule (SEDPP), and our newly proposed and more powerful rules - SSR-Dome and SSR-BEDPP.  
\item The whole algorithm is implemented in C++ and is optimized to be memory- and computation-efficient. Parallel computing via OpenMP is also supported wherever possible.
}

}

\note{
The input design matrix X must be a \code{\link[bigmemory]{big.matrix}} object. This can be created by the function \code{\link[bigmemory]{as.big.matrix}} in the R package \href{https://CRAN.R-project.org//package=bigmemory}{bigmemory}. If the data (design matrix) is very large (e.g. 10 GB) and stored in an external file, which is often the case for big data, X can be created by calling the function \code{\link{setupX}}. \strong{In this case, there are several restrictions about the data file:} 

\enumerate{
\item the data file must be a well-formated ASCII-file, with each row corresponding to an observation and each column a variable;
  \item the data file must contain only one single type. Current version only supports \code{double} type;
  \item the data file must contain only numeric variables. If there are categorical variables, the user needs to create dummy variables for each categorical varable (by adding additional columns).
}

Future versions will try to address these restrictions.

Denote the number of observations and variables be, respectively, \code{n} and \code{p}. It's worth noting that the package is more suitable for wide data (ultrahigh-dimensional, \code{p >> n}) as compared to long data (\code{n >> p}). This is because the model fitting algorithm takes advantage of sparsity assumption of high-dimensional data. To just give the user some ideas, below are some benchmarking results of the total computing time (in seconds) for solving lasso-penalized linear regression along a sequence of 100 values of the tuning parameter. In all cases, assume 20 non-zero coefficients equal +/- 2 in the true model. (Based on Version 1.2-3, screening rule "SSR-BEDPP" is used)

\itemize{
\item For wide data case (\code{p > n}), \code{n = 1,000}:
\tabular{ccccc}{
\code{p} \tab 1,000 \tab 10,000 \tab 100,000 \tab 1,000,000 \cr
Size of \code{X} \tab 9.5 MB \tab 95 MB \tab 950 MB \tab 9.5 GB \cr
Elapsed time (s) \tab 0.11 \tab 0.83 \tab 8.47 \tab 85.50 \cr
}
%\item For long data case (\code{n >> p}), \code{p = 1,000}:
%\tabular{ccccc}{
%\code{n} \tab 1,000 \tab 10,000 \tab 100,000 \tab 1,000,000 \cr
%Size of \code{X} \tab 9.5 MB \tab 95 MB \tab 950 MB \tab 9.5 GB \cr
%Elapsed time (s) \tab 2.50 \tab 11.43 \tab 83.69 \tab 1090.62 \cr
%}
}

}

\author{
  Yaohui Zeng and Patrick Breheny
  
  Maintainer: Yaohui Zeng <yaohui-zeng@uiowa.edu>
}

\keyword{package}

\examples{
\dontrun{
## Example of reading data from external big data file, fit lasso model, 
## and run cross validation in parallel

# simulated design matrix, 1000 observations, 500,000 variables, ~ 5GB
# there are 10 true variables with non-zero coefficient 2.
xfname <- 'x_e3_5e5.txt' 
yfname <- 'y_e3_5e5.txt' # response vector
time <- system.time(
  X <- setupX(xfname, sep = '\t') # create backing files (.bin, .desc)
)
print(time) # ~ 7 minutes; this is just one-time operation
dim(X)

# the big.matrix then can be retrieved by its descriptor file (.desc) in any new R session. 
rm(X)
xdesc <- 'x_e3_5e5.desc' 
X <- attach.big.matrix(xdesc)
dim(X)

y <- as.matrix(read.table(yfname, header = F))
time.fit <- system.time(
  fit <- biglasso(X, y, family = 'gaussian', screen = 'SSR-BEDPP')
)
print(time.fit) # ~ 44 seconds for fitting a lasso model along the entire solution path

# cross validation in parallel
seed <- 1234
time.cvfit <- system.time(
  cvfit <- cv.biglasso(X, y, family = 'gaussian', screen = 'SSR-BEDPP', 
                       seed = seed, ncores = 4, nfolds = 10)
)
print(time.cvfit) # ~ 3 minutes for 10-fold cross validation
plot(cvfit)
summary(cvfit)

}
}
