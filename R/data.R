#' Gene expression data from colon-cancer patients
#'
#' The data file contains gene expression data of 62 samples (40 tumor samples, 22 normal samples)
#' from colon-cancer patients analyzed with an Affymetrix oligonucleotide Hum6000 array.
#'
#' @format A list of 2 variables:
#'
#' \describe{
#'   \item{X}{A 62-by-2000 matrix that records the gene expression data. Used as design matrix.}
#'   \item{y}{
#'     A binary vector of length 62 recording the sample status: 1 = tumor; 0 = normal. Used as
#'     response vector.
#'   }
#' }
#'
#' @source The raw data can be found on Bioconductor:
#'   <https://bioconductor.org/packages/release/data/experiment/html/colonCA.html>.
#'
#' @references
#'   Alon U, Barkai N, Notterman DA, Gish K, Ybarra S, Mack D, and Levine AJ (1999). Broad patterns
#'   of gene expression revealed by clustering analysis of tumor and normal colon tissues probed by
#'   oligonucleotide arrays. *Proc. Natl. Acad. Sci.* 96: 6745--6750
#'   \doi{https://doi.org/10.1073/pnas.96.12.6745}
#'
#' @examples
#' data(colon)
#' X <- colon$X
#' y <- colon$y
#' str(X)
#' dim(X)
#' X.bm <- as.big.matrix(X, backingfile = "") # convert to big.matrix object
#' str(X.bm)
#' dim(X.bm)
"colon"
