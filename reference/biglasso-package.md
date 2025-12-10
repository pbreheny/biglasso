# Extending Lasso Model Fitting to Big Data

Extend lasso and elastic-net linear, logistic and cox regression models
for ultrahigh-dimensional, multi-gigabyte data sets that cannot be
loaded into available RAM. This package utilizes memory-mapped files to
store the massive data on the disk and only read those into memory
whenever necessary during model fitting. Moreover, some advanced feature
screening rules are proposed and implemented to accelerate the model
fitting. As a result, this package is much more memory- and
computation-efficient and highly scalable as compared to existing
lasso-fitting packages such as
[glmnet](https://CRAN.R-project.org/package=glmnet) and
[ncvreg](https://CRAN.R-project.org/package=ncvreg), thus allowing for
powerful big data analysis even with only an ordinary laptop.

## Details

|          |            |
|----------|------------|
| Package: | biglasso   |
| Type:    | Package    |
| Version: | 1.4-1      |
| Date:    | 2021-01-29 |
| License: | GPL-3      |

Penalized regression models, in particular the lasso, have been
extensively applied to analyzing high-dimensional data sets. However,
due to the memory limit, existing R packages are not capable of fitting
lasso models for ultrahigh-dimensional, multi-gigabyte data sets which
have been increasingly seen in many areas such as genetics, biomedical
imaging, genome sequencing and high-frequency finance.

This package aims to fill the gap by extending lasso model fitting to
Big Data in R. Version \>= 1.2-3 represents a major redesign where the
source code is converted into C++ (previously in C), and new feature
screening rules, as well as OpenMP parallel computing, are implemented.
Some key features of `biglasso` are summarized as below:

1.  it utilizes memory-mapped files to store the massive data on the
    disk, only loading data into memory when necessary during model
    fitting. Consequently, it's able to seamlessly data-larger-than-RAM
    cases.

2.  it is built upon pathwise coordinate descent algorithm with warm
    start, active set cycling, and feature screening strategies, which
    has been proven to be one of fastest lasso solvers.

3.  in incorporates our newly developed hybrid and adaptive screening
    that outperform state-of-the-art screening rules such as the
    sequential strong rule (SSR) and the sequential EDPP rule (SEDPP)
    with additional 1.5x to 4x speedup.

4.  the implementation is designed to be as memory-efficient as possible
    by eliminating extra copies of the data created by other R packages,
    making it at least 2x more memory-efficient than `glmnet`.

5.  the underlying computation is implemented in C++, and parallel
    computing with OpenMP is also supported.

**For more information:**

- Benchmarking results: <https://github.com/pbreheny/biglasso>

- Tutorial: <https://pbreheny.github.io/biglasso/articles/biglasso.html>

- Technical paper: <https://arxiv.org/abs/1701.05936>

## Note

The input design matrix X must be a
[`bigmemory::big.matrix()`](https://rdrr.io/pkg/bigmemory/man/big.matrix.html)
object. This can be created by the function `as.big.matrix` in the R
package [bigmemory](https://CRAN.R-project.org//package=bigmemory). If
the data (design matrix) is very large (e.g. 10 GB) and stored in an
external file, which is often the case for big data, X can be created by
calling the function
[`setupX()`](https://pbreheny.github.io/biglasso/reference/setupX.md).
**In this case, there are several restrictions about the data file:**

1.  the data file must be a well-formated ASCII-file, with each row
    corresponding to an observation and each column a variable;

2.  the data file must contain only one single type. Current version
    only supports `double` type;

3.  the data file must contain only numeric variables. If there are
    categorical variables, the user needs to create dummy variables for
    each categorical varable (by adding additional columns).

Future versions will try to address these restrictions.

Denote the number of observations and variables be, respectively, `n`
and `p`. It's worth noting that the package is more suitable for wide
data (ultrahigh-dimensional, `p >> n`) as compared to long data
(`n >> p`). This is because the model fitting algorithm takes advantage
of sparsity assumption of high-dimensional data. To just give the user
some ideas, below are some benchmarking results of the total computing
time (in seconds) for solving lasso-penalized linear regression along a
sequence of 100 values of the tuning parameter. In all cases, assume 20
non-zero coefficients equal +/- 2 in the true model. (Based on Version
1.2-3, screening rule "SSR-BEDPP" is used)

- For wide data case (`p > n`), `n = 1,000`:

  |                  |        |        |         |           |
  |------------------|--------|--------|---------|-----------|
  | `p`              | 1,000  | 10,000 | 100,000 | 1,000,000 |
  | Size of `X`      | 9.5 MB | 95 MB  | 950 MB  | 9.5 GB    |
  | Elapsed time (s) | 0.11   | 0.83   | 8.47    | 85.50     |

  %

- For long data case (`n >> p`), `p = 1,000`: %

  |                   |        |        |         |           |
  |-------------------|--------|--------|---------|-----------|
  | %`n`              | 1,000  | 10,000 | 100,000 | 1,000,000 |
  | %Size of `X`      | 9.5 MB | 95 MB  | 950 MB  | 9.5 GB    |
  | %Elapsed time (s) | 2.50   | 11.43  | 83.69   | 1090.62   |

## References

- Zeng Y and Breheny P. (2021) The biglasso Package: A Memory- and
  Computation-Efficient Solver for Lasso Model Fitting with Big Data
  in R. *R Journal*, **12**: 6-19.
  [doi:10.32614/RJ-2021-001](https://doi.org/10.32614/RJ-2021-001)

- Wang C and Breheny P. (2022) Adaptive hybrid screening for efficient
  lasso optimization. *Journal of Statistical Computation and
  Simulation*, **92**: 2233-2256.
  [doi:10.1080/00949655.2021.2025376](https://doi.org/10.1080/00949655.2021.2025376)

- Tibshirani, R., Bien, J., Friedman, J., Hastie, T., Simon, N., Taylor,
  J., and Tibshirani, R. J. (2012). Strong rules for discarding
  predictors in lasso-type problems. *Journal of the Royal Statistical
  Society: Series B (Statistical Methodology)*, **74**(2), 245-266.

- Wang, J., Zhou, J., Wonka, P., and Ye, J. (2013). Lasso screening
  rules via dual polytope projection. *In Advances in Neural Information
  Processing Systems*, pp. 1070-1078.

- Xiang, Z. J., and Ramadge, P. J. (2012). Fast lasso screening tests
  based on correlations. *In Acoustics, Speech and Signal Processing
  (ICASSP), 2012 IEEE International Conference on* (pp. 2137-2140).
  IEEE.

- Wang, J., Zhou, J., Liu, J., Wonka, P., and Ye, J. (2014). A safe
  screening rule for sparse logistic regression. *In Advances in Neural
  Information Processing Systems*, pp. 1053-1061.

## Author

Yaohui Zeng, Chuyi Wang, Tabitha Peter, and Patrick Breheny

## Examples

``` r
if (FALSE) { # \dontrun{
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
  fit <- biglasso(X, y, family = 'gaussian', screen = 'Hybrid')
)
print(time.fit) # ~ 44 seconds for fitting a lasso model along the entire solution path

# cross validation in parallel
seed <- 1234
time.cvfit <- system.time(
  cvfit <- cv.biglasso(X, y, family = 'gaussian', screen = 'Hybrid', 
                       seed = seed, ncores = 4, nfolds = 10)
)
print(time.cvfit) # ~ 3 minutes for 10-fold cross validation
plot(cvfit)
summary(cvfit)
} # }
```
