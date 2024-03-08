# TKP 
# March 2024
# Working to debug issues in gaussian_simple.cpp 
#   let's examine each helper function 

data(colon)
X <- colon$X
y <- colon$y
X.bm <- as.big.matrix(X)

# get_residual ------------------------------------ 
n <- nrow(X)
p <- ncol(X)
col_idx <- 1:p
z <- rep(0, length(p))
lambda <- 0.1
xmax_ptr <- 0
XMat <- X.bm
row_idx <- 1:nrow(X)
alpha <- 1
# res <- .Call("get_residual",
#              col_idx,
#              z,
#              lambda,
#              xmax_ptr,
#              XMat,
#              y,
#              row_idx,
#              alpha,
#              n,
#              p,
#              PACKAGE = 'biglasso')
# sum -----------------------------------------------

# crossprod_resid_no_std ------------------------------

# pow ------------------------------------------------


# update_resid_no_std -------------------------------


