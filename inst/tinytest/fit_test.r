devtools::load_all('.')
data(colon)
X <- colon$X[, 1:20] |> ncvreg::std()
xtx <- apply(X, 2, crossprod)
y <- colon$y
X.bm <- as.big.matrix(X)
fit_flex <- biglasso_fit(X.bm, y, lambda = 0.01, xtx=xtx)

#plot(fit_flex, log.l = TRUE, main = 'lasso')
