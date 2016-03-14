library(testthat)
library(biglasso)

context("Testing biglasso(): ")

context("Linear regression against ncvreg:")

## Linear regression
data(prostate)
X <- as.matrix(prostate[,1:8])
y <- prostate$lpsa
X.bm <- as.big.matrix(X)

fit1 <- biglasso(X.bm, y, family = 'gaussian', penalty = 'lasso')
fit2 <- ncvreg(X, y, family = 'gaussian', penalty = 'lasso', returnX = TRUE)

teest_that("lambda's are equal: ",{
  expect_equal(fit1$lambda, fit2$lambda)
})
