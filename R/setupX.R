## read and set up design matrix X from external ASCII-file
setupX <- function(filename, dir = getwd(), sep = ",", 
                   backingfile = paste0(unlist(strsplit(filename, 
                                                        split = "\\."))[1], 
                                        ".bin"),
                   descriptorfile = paste0(unlist(strsplit(filename, 
                                                           split = "\\."))[1], 
                                           ".desc"), 
                   ...) {
  # create file backing cache
  cat("Reading data from file, and creating file-backed big.matrix...\n")
  cat("This should take a while if the data is very large...\n")
  cat("Start time: ", format(Sys.time()), "\n")
  dat <- read.big.matrix(filename = filename, sep = sep, type = 'double', 
                         separated = FALSE, 
                         backingfile = backingfile, descriptorfile = descriptorfile,
                         backingpath = dir, shared = TRUE, ...)
  cat("End time: ", format(Sys.time()), "\n")
  cat("DONE!\n")
  
  rm(dat)
  gc()
  
  ## attach the descriptor information as the reference of the big.matrix
  X <- attach.big.matrix(descriptorfile, backingpath = dir)
  X
}

# # # ## test
# setwd("~/GitHub/bigLasso_wide/")
# require(bigmemory)
# require(pryr)
# mem_used()
# filename <- "test_setupX.txt"
# X <- setupX(filename, sep = "\t")
# mem_used()
# object_size(X)
# dim(X)
# X[1:10, 1:10]
