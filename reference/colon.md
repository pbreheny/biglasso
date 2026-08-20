# Gene expression data from colon-cancer patients

The data file contains gene expression data of 62 samples (40 tumor
samples, 22 normal samples) from colon-cancer patients analyzed with an
Affymetrix oligonucleotide Hum6000 array.

## Usage

``` r
data(colon)
```

## Format

A list of 2 variables:

- X:

  A 62-by-2000 matrix that records the gene expression data. Used as
  design matrix.

- y:

  A binary vector of length 62 recording the sample status: 1 = tumor; 0
  = normal. Used as response vector.

## Source

The raw data can be found on Bioconductor:
<https://bioconductor.org/packages/release/data/experiment/html/colonCA.html>.

## References

Alon U, Barkai N, Notterman DA, Gish K, Ybarra S, Mack D, and Levine AJ
(1999). Broad patterns of gene expression revealed by clustering
analysis of tumor and normal colon tissues probed by oligonucleotide
arrays. *Proc. Natl. Acad. Sci.* 96: 6745–6750
[doi:10.1073/pnas.96.12.6745](https://doi.org/10.1073/pnas.96.12.6745)

## Examples

``` r
data(colon)
X <- colon$X
y <- colon$y
str(X)
#>  num [1:62, 1:2000] 8589 9164 3826 6246 3230 ...
#>  - attr(*, "dimnames")=List of 2
#>   ..$ : chr [1:62] "t" "n" "t" "n" ...
#>   ..$ : chr [1:2000] "Hsa.3004" "Hsa.13491" "Hsa.13491.1" "Hsa.37254" ...
dim(X)
#> [1]   62 2000
X.bm <- as.big.matrix(X, backingfile = "") # convert to big.matrix object
str(X.bm)
#> Formal class 'big.matrix' [package "bigmemory"] with 1 slot
#>   ..@ address:<pointer: 0x55b0ee00b690> 
dim(X.bm)
#> [1]   62 2000
```
