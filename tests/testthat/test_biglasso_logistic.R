library(testthat)
library(biglasso)
library(glmnet)

context("Testing logistic regression:")

set.seed(1234)
n <- 100
p <- 10
eps <- 1e-12
tolerance <- 1e-3
X <- matrix(rnorm(n*p), n, p)
b <- rnorm(p)

y <- rbinom(n, 1, prob = exp(1 + X %*% b) / (1 + exp(1 + X %*% b)))
fit.mle <- glm(y ~ X, family = 'binomial')
beta <- fit.mle$coefficients

glmnet.control(fdev = 0, devmax = 1)
fit.glm <- glmnet(X, y, family = 'binomial', thresh = eps, lambda.min.ratio = 0)

X.bm <- as.big.matrix(X)
fit.ssr <- biglasso(X.bm, y, family = 'binomial', eps = eps, lambda.min = 0)
fit.ssr.mm <- biglasso(X.bm, y, family = 'binomial', eps = eps, alg.logistic = 'MM', lambda.min = 0)

test_that("Test against MLE: ",{
  expect_equal(as.numeric(beta), as.numeric(fit.ssr$beta[, 100]), tolerance = tolerance)
  expect_equal(as.numeric(fit.ssr$beta[, 100]), as.numeric(fit.ssr.mm$beta[, 100]), tolerance = tolerance)
})

test_that("Test against glmnet: ",{
  expect_equal(as.numeric(fit.glm$beta), as.numeric(fit.ssr$beta[-1, ]), tolerance = tolerance)
  expect_equal(as.numeric(fit.glm$beta), as.numeric(fit.ssr.mm$beta[-1, ]), tolerance = tolerance)
})
