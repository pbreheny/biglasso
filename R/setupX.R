#' Set up design matrix X by reading data from big data file
#' 
#' Set up the design matrix X as a `big.matrix` object based on external
#' massive data file stored on disk that cannot be fullly loaded into memory.
#' The data file must be a well-formated ASCII-file, and contains only one
#' single type. Current version only supports `double` type. Other
#' restrictions about the data file are described in
#' [biglasso-package]. This function reads the massive data, and
#' creates a `big.matrix` object. By default, the resulting
#' `big.matrix` is file-backed, and can be shared across processors or
#' nodes of a cluster.
#' 
#' For a data set, this function needs to be called only one time to set up the
#' `big.matrix` object with two backing files (.bin, .desc) created in
#' current working directory. Once set up, the data can be "loaded" into any
#' (new) R session by calling `attach.big.matrix(discriptorfile)`.
#' 
#' This function is a simple wrapper of [bigmemory::read.big.matrix()]. See
#' \href{https://CRAN.R-project.org/package=bigmemory}{bigmemory} for more
#' details.
#' 
#' @param filename The name of the data file. For example, "dat.txt".
#' @param dir The directory used to store the binary and descriptor files
#' associated with the `big.matrix`. The default is current working
#' directory.
#' @param sep The field separator character. For example, "," for
#' comma-delimited files (the default); "\\t" for tab-delimited files.
#' @param backingfile The binary file associated with the file-backed
#' `big.matrix`. By default, its name is the same as `filename` with
#' the extension replaced by ".bin".
#' @param descriptorfile The descriptor file used for the description of the
#' file-backed `big.matrix`. By default, its name is the same as
#' `filename` with the extension replaced by ".desc".
#' @param type The data type. Only "double" is supported for now.
#' @param ... Additional arguments that can be passed into function
#' [bigmemory::read.big.matrix()].
#' 
#' @returns A `big.matrix` object corresponding to a file-backed
#' [bigmemory::big.matrix()]. It's ready to be used as the design matrix `X` in
#' [biglasso()] and [cv.biglasso()].
#' 
#' @author Yaohui Zeng and Patrick Breheny
#' 
#' @seealso [biglasso()], [ncvreg::cv.ncvreg()], [biglasso-package]
#' 
#' @examples
#' ## see the example in "biglasso-package"
#' @export

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
