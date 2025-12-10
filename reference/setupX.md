# Set up design matrix X by reading data from big data file

Set up the design matrix X as a `big.matrix` object based on external
massive data file stored on disk that cannot be fullly loaded into
memory. The data file must be a well-formated ASCII-file, and contains
only one single type. Current version only supports `double` type. Other
restrictions about the data file are described in
[biglasso-package](https://pbreheny.github.io/biglasso/reference/biglasso-package.md).
This function reads the massive data, and creates a `big.matrix` object.
By default, the resulting `big.matrix` is file-backed, and can be shared
across processors or nodes of a cluster.

## Usage

``` r
setupX(
  filename,
  dir = getwd(),
  sep = ",",
  backingfile = paste0(unlist(strsplit(filename, split = "\\."))[1], ".bin"),
  descriptorfile = paste0(unlist(strsplit(filename, split = "\\."))[1], ".desc"),
  type = "double",
  ...
)
```

## Arguments

- filename:

  The name of the data file. For example, "dat.txt".

- dir:

  The directory used to store the binary and descriptor files associated
  with the `big.matrix`. The default is current working directory.

- sep:

  The field separator character. For example, "," for comma-delimited
  files (the default); "\t" for tab-delimited files.

- backingfile:

  The binary file associated with the file-backed `big.matrix`. By
  default, its name is the same as `filename` with the extension
  replaced by ".bin".

- descriptorfile:

  The descriptor file used for the description of the file-backed
  `big.matrix`. By default, its name is the same as `filename` with the
  extension replaced by ".desc".

- type:

  The data type. Only "double" is supported for now.

- ...:

  Additional arguments that can be passed into function
  [`bigmemory::read.big.matrix()`](https://rdrr.io/pkg/bigmemory/man/write.big.matrix.html).

## Value

A `big.matrix` object corresponding to a file-backed
[`bigmemory::big.matrix()`](https://rdrr.io/pkg/bigmemory/man/big.matrix.html).
It's ready to be used as the design matrix `X` in
[`biglasso()`](https://pbreheny.github.io/biglasso/reference/biglasso.md)
and
[`cv.biglasso()`](https://pbreheny.github.io/biglasso/reference/cv.biglasso.md).

## Details

For a data set, this function needs to be called only one time to set up
the `big.matrix` object with two backing files (.bin, .desc) created in
current working directory. Once set up, the data can be "loaded" into
any (new) R session by calling `attach.big.matrix(discriptorfile)`.

This function is a simple wrapper of
[`bigmemory::read.big.matrix()`](https://rdrr.io/pkg/bigmemory/man/write.big.matrix.html).
See [bigmemory](https://CRAN.R-project.org/package=bigmemory) for more
details.

## See also

[`biglasso()`](https://pbreheny.github.io/biglasso/reference/biglasso.md),
[`ncvreg::cv.ncvreg()`](https://pbreheny.github.io/ncvreg/reference/cv.ncvreg.html),
[biglasso-package](https://pbreheny.github.io/biglasso/reference/biglasso-package.md)

## Author

Yaohui Zeng and Patrick Breheny

## Examples

``` r
## see the example in "biglasso-package"
```
