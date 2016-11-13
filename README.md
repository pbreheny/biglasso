
# Big Lasso: Extend Lasso Model Fitting to Big Data in R

[![Build Status](https://travis-ci.org/YaohuiZeng/biglasso.svg?branch=master)](https://travis-ci.org/YaohuiZeng/biglasso)
[![CRAN_Status_Badge](http://www.r-pkg.org/badges/version/biglasso)](https://CRAN.R-project.org/package=biglasso)
[![CRAN RStudio mirror downloads](http://cranlogs.r-pkg.org/badges/grand-total/biglasso)](http://www.r-pkg.org/pkg/biglasso)

`biglasso` extends lasso and elastic-net model fitting for ultrahigh-dimensional, multi-gigabyte 
data sets that cannot be loaded into memory. It utilizes memory-mapped files to store the massive data on the disk and only read those into memory whenever necessary during model fitting. Moreover, some advanced feature screening rules are proposed and implemented to accelerate the model fitting. As a result, this package is much more memory- and computation-efficient as compared to existing lasso-fitting packages such as [glmnet](https://CRAN.R-project.org/package=glmnet) and [ncvreg](https://CRAN.R-project.org/package=ncvreg), thus allowing for very powerful big data analysis even with only an ordinary laptop.


## Features:
1. It utilizes memory-mapping files to store the massive data on the disk and only reads those into memory whenever necessary during model fitting, thus can easily handle data-larger-than-RAM cases.
2. It builds upon the pathwise coordinate descent algorithm combined with "warm start", "active set cycling", and feature screening strategies, which has been proved to be one of the most powerful lasso solvers.
3. It incorporates some efficient feature screening rules, such as the sequential strong rule (SSR), the sequential EDPP rule (SEDPP), and our newly proposed and more powerful rules - SSR-Dome and SSR-BEDPP.  
4. The underlying algorithm is implemented in C++ and is optimized to be memory- and computation-efficient. Parallel computing via OpenMP is also supported.

## Installation:
* The stable version: `install.packages("biglasso")`
* The latest version: `devtools::install_github("YaohuiZeng/biglasso")`


## Report bugs：
* open an [issue](https://github.com/YaohuiZeng/biglasso/issues) or send an email to Yaohui Zeng at <yaohui-zeng@uiowa.edu>


## News:
* This package on GitHub has been updated to Version 1.2-3. See details in NEWS.
* The newest stable version will be submitted to CRAN soon after testing.
