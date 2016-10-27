## read and set up design matrix X from external ASCII-file
setupX <- function(filename, dir = getwd(), sep = ",", 
                   backingfile = paste0(unlist(strsplit(filename, 
                                                        split = "\\."))[1], 
                                        ".bin"),
                   descriptorfile = paste0(unlist(strsplit(filename, 
                                                           split = "\\."))[1], 
                                           ".desc"), 
                   type = 'double',
                   ...) {
  
  # create file backing cache
  cat("Reading data from file, and creating file-backed big.matrix...\n")
  cat("This should take a while if the data is very large...\n")
  cat("Start time: ", format(Sys.time()), "\n")
  dat <- read.big.matrix(filename = filename, sep = sep, type = type,
                         separated = FALSE, 
                         backingfile = backingfile, descriptorfile = descriptorfile,
                         backingpath = dir, shared = TRUE, ...)
  cat("End time: ", format(Sys.time()), "\n")
  cat("DONE!\n\n")
  cat("Note: This function needs to be called only one time to create two backing\n")
  cat("      files (.bin, .desc) in current dir. Once done, the data can be\n")
  cat("      'loaded' using function 'attach.big.matrix'. See details in doc. \n")
  
  rm(dat)
  gc()
  
  ## attach the descriptor information as the reference of the big.matrix
  X <- attach.big.matrix(descriptorfile, backingpath = dir)
  X

}
