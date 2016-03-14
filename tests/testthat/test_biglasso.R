
library(testthat)
library(biglasso)
# 
# require(bigmemory)
# require(Rcpp)
# require(RcppArmadillo)
# require(Matrix)
# require(ncvreg)
# 
# dyn.load("~/GitHub/biglasso/src/biglasso.so")
# ## import R functions
# source("~/GitHub/biglasso/R/biglasso.R")
# source("~/GitHub/biglasso/R/predict.R")
# source("~/GitHub/biglasso/R/loss.R")
# source("~/GitHub/biglasso/R/cv.biglasso.R")
# source("~/GitHub/biglasso/R/plot.biglasso.R")
# source("~/GitHub/biglasso/R/setupLambda.R")

context("Testing linear regression: biglasso against ncvreg:")

## Linear regression
seed <- 1234
data(prostate)
X <- as.matrix(prostate[,1:8])
y <- prostate$lpsa
X.bm <- as.big.matrix(X)
fit1 <- biglasso(X.bm, y, family = 'gaussian', penalty = 'lasso')
fit2 <- ncvreg(X, y, family = 'gaussian', penalty = 'lasso', returnX = TRUE)
cvfit1 <- cv.biglasso(X.bm, y, family = 'gaussian', seed = seed)
cvfit2 <- cv.ncvreg(X, y, family = 'gaussian', penalty = 'lasso', seed = seed)
# cvfit3 <- cv.biglasso(X.bm, y, family = 'gaussian', seed = seed, ncores = 2)

test_that("Test beta: ",{
  expect_equal(fit1$beta, Matrix(fit2$beta, sparse = T, dimnames = NULL))
  expect_equal(dimnames(fit1$beta), dimnames(fit2$beta))
})

test_that("Test lambda, loss, center, scale: ",{
  expect_equal(fit1$lambda, fit2$lambda)
  expect_equal(fit1$loss, fit2$loss)
  expect_equal(fit1$center, fit2$center)
  expect_equal(fit1$scale, fit2$scale)
})

test_that("Test prediction:",{
  expect_equal(predict(fit1, X = X.bm, type = 'coefficients'), 
               Matrix(predict(fit2, X, type = 'coefficients'), sparse = T))
  expect_equal(predict(fit1, X = X.bm, type = 'nvars'), predict(fit2, X, type = 'nvars'))
  expect_equal(predict(fit1, X = X.bm, type = 'vars'), predict(fit2, X, type = 'vars'))
  expect_equal(predict(fit1, X = X.bm, type = 'link'), 
               Matrix(predict(fit2, X, type = 'link')))
})

test_that("Test cross-validation: ",{
  expect_equal(cvfit1$cve, cvfit2$cve)
  expect_equal(cvfit1$cvse, cvfit2$cvse)
  expect_equal(cvfit1$min, cvfit2$min)
  expect_equal(cvfit1$lambda.min, cvfit2$lambda.min)
})

test_that("Test parallel cross-validation: ",{
  expect_equal(cvfit1$cve, cvfit3$cve)
  expect_equal(cvfit1$cvse, cvfit3$cvse)
  expect_equal(cvfit1$min, cvfit3$min)
  expect_equal(cvfit1$lambda.min, cvfit3$lambda.min)
})




