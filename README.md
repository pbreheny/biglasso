

[![Build Status](https://travis-ci.org/YaohuiZeng/biglasso.svg?branch=master)](https://travis-ci.org/YaohuiZeng/biglasso)
[![CRAN_Status_Badge](http://www.r-pkg.org/badges/version/biglasso)](http://cran.r-project.org/package=biglasso)


# Big Lasso: Extending Lasso Model Fitting to Big Data in R

`biglasso` Extend lasso and elastic-net model fitting for ultrahigh-dimensional, multi-gigabyte 
data sets that cannot be loaded into memory. Compared to existing lasso-fitting packages, 
it preserves equivalently fast computation speed but is much more memory-efficient, 
thus allowing for very powerful big data analysis even with only a single laptop.

To install:
* the stable version: `install.packages("biglasso")`
* the latest version (requires devtools): `install_github("YaohuiZeng/biglasso")`

To report bugs：
* send email to Yaohui Zeng at <yaohui-zeng@uiowa.edu>


Note:

This package is under heavy development. In the newest version, I have implemented several sparse screening rules along with parallel computing and better algorithm design to speed up the computation. 

Several users have already reported that there are some issues on Windows. I will do more testing on different OS for the next version, which is expected to be released by end of August. If you encounter any issues or questions, please don't hesitate to send me an email.
