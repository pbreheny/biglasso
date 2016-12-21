
# biglasso: Extend Lasso Model Fitting to Big Data in R

[![Build Status](https://travis-ci.org/YaohuiZeng/biglasso.svg?branch=master)](https://travis-ci.org/YaohuiZeng/biglasso)
[![CRAN_Status_Badge](http://www.r-pkg.org/badges/version/biglasso)](https://CRAN.R-project.org/package=biglasso)
[![CRAN RStudio mirror downloads](http://cranlogs.r-pkg.org/badges/grand-total/biglasso)](http://www.r-pkg.org/pkg/biglasso)

`biglasso` extends lasso and elastic-net linear and logistic regression models for ultrahigh-dimensional, multi-gigabyte data sets that cannot be loaded into memory. It utilizes memory-mapped files to store the massive data on the disk and only read those into memory whenever necessary during model fitting. Moreover, some advanced feature screening rules are proposed and implemented to accelerate the model fitting. **As a result, this package is much more memory- and computation-efficient and highly scalable as compared to existing lasso-fitting packages such as [glmnet](https://CRAN.R-project.org/package=glmnet) and [ncvreg](https://CRAN.R-project.org/package=ncvreg)**. Bechmarking experiments using both simulated and real data sets show that `biglasso` is not only 1.5x to 4x times faster than existing packages, but also at least 2x more memory-efficient. More importantly, to the best of our knowledge, `biglasso` is the first R package that enables users to fit lasso models with data sets that are larger than available RAM, thus allowing for powerful big data analysis on an ordinary laptop.


## Features:
1. It utilizes memory-mapped files to store the massive data on the disk, only loading data into memory when necessary during model fitting. Consequently, it's able to seamlessly handle out-of-core computation.
2. It is built upon pathwise coordinate descent algorithm with *warm start, active set cycling, and feature screening* strategies, which has been proven to be one of fastest lasso solvers.
3. We develop new, hybrid feature screening rules that outperform state-of-the-art screening rules such as the sequential strong rule (SSR) and the sequential EDPP rule (SEDPP) with additional 1.5x to 4x speedup.
4. The implementation is designed to be as memory-efficient as possible by eliminating extra copies of the data created by other R packages, making `biglasso` at least 2x more memory-efficient than `glmnet`.
5. The underlying computation is implemented in C++, and parallel computing with OpenMP is also supported.


## Benchmarks:

### Simulated data:

* **Packages** to be compared: `biglasso (1.2-3)` (with `screen = "SSR-BEDPP"`), `glmnet (2.0-5)`, `ncvreg (3.7-0)`, and `picasso (0.5-4)`. 
* **Platform**: MacBook Pro with Intel Core i7 @ 2.3 GHz and 16 GB RAM.
* **Experiments**: solving lasso-penalized linear regression over the entire path of 100 $\lambda$ values equally spaced on the scale of `lambda / lambda_max` from 0.1 to 1; varying number of observations `n` and number of features `p`; 20 replications, the mean (SE) computing time (in seconds) are reported.
* **Data generating model**: `y =  X *  beta + 0.1 eps`, where `X` and `eps` are i.i.d. sampled from `N(0, 1)`.

<!---
![Alt text](/vignettes/2016-11-04_vary_p_pkgs_2.png?raw=true "Vary p")![Alt text](/vignettes/2016-11-05_vary_n_pkgs_2.png?raw=true "Vary n")
-->


#### (1) `biglasso` is more computation-efficient:

<img src="/vignettes/2016-11-20_vary_p_pkgs.png" width="400" height="300" /><img src="/vignettes/2016-11-20_vary_n_pkgs.png" width="400" height="300" />

In all the settings, `biglasso` (1 core) is uniformly 2x faster than `glmnet` and `ncvreg`, and 2.5x faster than `picasso`. Moreover, the computing time of `biglasso` can be further reduced by half via parallel-computation of 4 cores.

#### (2) `biglasso` is more memory-efficient:


To prove that `biglasso` is much more memory-efficient, we simulate a `1000 X 100000` large feature matrix. The raw data is 0.75 GB. We used [Syrupy](https://github.com/jeetsukumaran/Syrupy) to measure the memory used in RAM (i.e. the resident set size, RSS) every 1 second during lasso model fitting by each of the packages. 

The maximum RSS (in **GB**) used by a single fit and 10-fold cross validation is reported in the Table below. In the single fit case, `biglasso` consumes 0.84 GB memory in RAM, 50% of that used by `glmnet` and  22% of that used by `picasso`. Note that the memory consumed by `glmnet`, `ncvreg`, and `picasso` are respectively 2.2x, 2.1x, and 5.1x larger than the size of the raw data. More strikingly, `biglasso` does not require additional memory to perform cross-validation, unlike other packages.  For serial 10-fold cross-validation, `biglasso`  requires just 27% of the memory used by `glmnet` and 23% of that used by `ncvreg`, making it 3.6x and 4.3x more memory-efficient compared to these two, respectively.

<center>

|   Package  |  picasso |  ncvreg  |  glmnet  |  biglasso  |
|-----------:|:--------:|:--------:|:--------:|:----------:|
| Single fit |   3.84   |   1.60   |   1.67   |    0.84    | 
| 10-fold CV |    -     |   3.74   |   3.18   |    0.87    |

</center>

**Note**:
..* the memory savings offered by `biglasso` would be even more significant if cross-validation were conducted in parallel. However, measuring memory usage across parallel processes is not straightforward and not implemented in `Syrupy`;
..* cross-validation is not implemented in `picasso` at this point.


### Real data:

The performance of the packages are also tested using diverse real data sets: 
* [Breast cancer gene expression data](http://myweb.uiowa.edu/pbreheny/data/bcTCGA.html) (GENE); 
* [MNIST handwritten image data](http://yann.lecun.com/exdb/mnist/) (MNIST);
* [Cardiac fibrosis genome-wide association study data](https://arxiv.org/abs/1607.05636) (GWAS);
* [Subset of New York Times bag-of-words data](https://archive.ics.uci.edu/ml/datasets/Bag+of+Words) (NYT).

The following table summarizes the mean (SE) computing time (in seconds) of solving the lasso along the entire path of 100 `lambda` values equally spaced on the scale of `lambda / lambda_max` from 0.1 to 1 over 20 replications.

<center>

| Package |     GENE    |    MNIST    |      GWAS    |      NYT     |
|--------:|:-----------:|:-----------:|:------------:|:------------:|
|         |   `n=536`   |   `n=784`   |    `n=313`   |   `n=5,000`  | 
|         | `p=17,322`  |  `p=60,000` |  `p=660,495` |  `p=55,000`  |
| picasso | 1.50 (0.01) | 6.86 (0.06) | 34.00 (0.47) | 44.24 (0.46) |
| ncvreg  | 1.14 (0.02) | 5.60 (0.06) | 31.55 (0.18) | 32.78 (0.10) |
| glmnet  | 1.02 (0.01) | 5.63 (0.05) | 23.23 (0.19) | 33.38 (0.08) |
|biglasso | 0.54 (0.01) | 1.48 (0.10) | 17.17 (0.11) | 14.35 (1.29) |

</center>


### Big data: Out-of-core computation

To demonstrate the out-of-core computing capability of `biglasso`, a 31 GB real data set from a large-scale genome-wide association study is analyzed. The dimensionality of the design matrix is: `n = 2898, p = 1,339,511`. **Note that the size of data is nearly 2x larger than the installed 16 GB of RAM.**

Since other three packages cannot handle this data-larger-than-RAM case, we compare the performance of screening rules `SSR` and `SSR-BEDPP` based on our package `biglasso`. In addition, two cases in terms of `lambda_min` are considered: (1) `lam_min = 0.1 lam_max`; and (2) `lam_min = 0.5 lam_max`, as in practice there is typically less interest in lower values of `lambda`for very high-dimensional data such as this case. Again the entire solution path with 100 `lambda` values is obtained. The table below summarizes the overall computing time (in **minutes**) by screening rule ``SSR`` (which is what other three packages are using) and our new rule ``SSR-BEDPP``. (Only 1 trial is conducted.)

|               Cases                |   SSR  |  SSR-BEDP  |
|:-----------------------------------|-------:|-----------:|
| `lam_min / lam_max = 0.1`, 1 core  | 284.56 |   189.21   | 
| `lam_min / lam_max = 0.1`, 4 cores | 142.55 |    93.74   |
| `lam_min / lam_max = 0.5`, 1 core  | 285.61 |   102.75   | 
| `lam_min / lam_max = 0.5`, 4 cores | 141.28 |    51.02   |


## Installation:
* The stable version: `install.packages("biglasso")`
* The latest version: `devtools::install_github("YaohuiZeng/biglasso")`


## Report bugs：
* open an [issue](https://github.com/YaohuiZeng/biglasso/issues) or send an email to Yaohui Zeng at <yaohui-zeng@uiowa.edu>


## News:
* This package on GitHub has been updated to Version 1.2-5. See details in NEWS.
* The newest stable version will be submitted to CRAN soon after testing.
