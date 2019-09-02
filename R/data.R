#' Gene expression data from colon-cancer patients
#' 
#' The data file contains gene expression data of 62 samples (40 tumor samples,
#' 22 normal samples) from colon-cancer patients analyzed with an Affymetrix
#' oligonucleotide Hum6000 array.
#' 
#' @name colon
#' @docType data
#' @format A list of 2 variables included in \code{colon}: \itemize{
#' \item\code{X}: a 62-by-2000 matrix that records the gene expression data.
#' Used as design matrix.
#' \item\code{y}: a binary vector of length 62 recording the sample status: 1 =
#' tumor; 0 = normal. Used as response vector.
#' }
#' @references \itemize{ \item U. Alon et al. (1999): Broad patterns of gene
#' expression revealed by clustering analysis of tumor and normal colon tissue
#' probed by oligonucleotide arrays. \emph{Proc. Natl. Acad. Sci. USA}
#' \strong{96}, 6745-6750. \url{http://www.pnas.org/content/96/12/6745.short}.
#' }
#' @source The raw data can be found on Bioconductor:
#' \url{https://bioconductor.org/packages/release/data/experiment/html/colonCA.html}.
#' @keywords datasets
#' @examples
#' data(colon)
#' X <- colon$X
#' y <- colon$y
#' str(X)
#' dim(X)
#' X.bm <- as.big.matrix(X, backingfile = "") # convert to big.matrix object
#' str(X.bm)
#' dim(X.bm)
NULL