devtools::load_all('.')
data(colon)
X <- colon$X[, 1:20] |> ncvreg::std()
xtx <- apply(X, 2, crossprod)
init <- rep(0, ncol(X))
resid <- drop(y - X %*% init)
y <- colon$y
X.bm <- as.big.matrix(X)
fit_flex <- biglasso_fit(X.bm, y, lambda = 0.01, xtx=xtx, r = resid)

#plot(fit_flex, log.l = TRUE, main = 'lasso')
