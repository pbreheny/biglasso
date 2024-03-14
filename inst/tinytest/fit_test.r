devtools::load_all('.')
data(colon)
X <- colon$X[, 1:20] |> ncvreg::std()
# X <- cbind(1, X)
xtx <- apply(X, 2, crossprod)
init <- rep(0, ncol(X))
y <- colon$y
resid <- drop(y - X %*% init)
X.bm <- as.big.matrix(X)

fit1 <- biglasso_fit(X.bm, y, lambda = 0.01, xtx=xtx, r = resid)
fit1$beta

# TODO: figure out how to adapt plot function to work with biglasso_fit()
# plot(fit_flex, log.l = TRUE, main = 'lasso') 

# compare with `ncvreg::ncvfit()`
library(ncvreg)
fit2 <- ncvfit(X = X, y = y, lambda = 0.01, xtx = xtx, r = resid, penalty = "lasso")
fit2$beta

# these should be the same! 
tinytest::expect_equal(fit1$beta, fit2$beta, tolerance = 0.1)
tinytest::expect_equal(fit1$beta, fit2$beta, tolerance = 0.01)

# look at these values
data.frame(biglasso = fit1$beta,
           ncvfit = fit2$beta) |>
  round(digits = 4) |> 
  View()
