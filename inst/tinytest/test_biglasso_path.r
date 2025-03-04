if (interactive()) library(tinytest)
library(ncvreg)
library(glmnet)

# colon data ------------------------------
data(colon)
X <- colon$X |> ncvreg::std()
X <- cbind(1, X)
xtx <- apply(X, 2, crossprod)
init <- rep(0, ncol(X)) # cold starts - use more iterations (default is 1000)
y <- colon$y
og_resid <- resid <- drop(y - X %*% init)
og_X <- X.bm <- as.big.matrix(X)
n <- nrow(X)
p <- ncol(X)

# choose a lambda path of just 10 values, for sake of testing 
lam <- ncvreg:::setupLambda(X = X, y = y, family = "gaussian", nlambda = 10,
                            penalty.factor = rep(1, ncol(X)),
                            alpha = 1,
                            lambda.min = ifelse(n > p, 0.001, 0.05))

fit1 <- biglasso_path(X.bm, y, lambda = lam, xtx=xtx, r = resid,
                     penalty = "lasso", max.iter = 20000)


fit2 <- glmnet(X, y = y, family = "gaussian", lambda = lam, 
               penalty.factor = rep(1, ncol(X)),
               penalty = "lasso", max.iter = 10000, standardize = F)

expect_equivalent(fit1$beta, fit2$beta, tolerance = 0.001)


# prostate data ---------------------------
data(Prostate)
X <- Prostate$X |> ncvreg::std()
X <- cbind(1, X)
xtx <- apply(X, 2, crossprod)
init <- rep(0, ncol(X)) # cold starts - use more iterations (default is 1000)
y <- Prostate$y
og_resid <- resid <- drop(y - X %*% init)
og_X <- X.bm <- as.big.matrix(X)
n <- nrow(X)
p <- ncol(X)

fit3 <- biglasso_path(X.bm, y, lambda = lam, xtx=xtx, r = resid,
                      penalty = "lasso", max.iter = 10000)


fit4 <- glmnet(X, y = y, family = "gaussian", lambda = lam, 
               penalty.factor = rep(1, ncol(X)),
               penalty = "lasso", max.iter = 10000, standardize = F)

expect_equivalent(fit3$beta, fit4$beta, tolerance = 0.001)
