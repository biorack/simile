# SIMILE

SIMILE (Significant Interrelation of MS/MS Ions via Laplacian Embedding) is a Python library for pairwise alignment of fragmentation spectra with significance estimation and is robust to multiple differences in chemical structure.  
[bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2021.02.24.432767v1)

![SIMILE Flow](SimileFig1Vert.png "SIMILE")

## Features
- Generate substitution matrices interrelating fragment ions in fragmention spectra  
(Just like how substitution matrices interrelate amino acids in protein sequences!)

- Align and score fragmentation spectra according to the substitutability of their fragment ions

- Calculate the significance of aligned fragmentation spectra

- BONUS: Less than 200 lines of intelligible code!

## Installation

Use the package manager [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/index.html) to install environment.yml for minimum requirements. Alternatively, use eniviroment-example.yml to run the example notebook.

```bash
conda env create -f environment-base.yml
```

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mwang87/simile/HEAD)

## Python dependencies
- python3 (pinned to 3.7 currently due to non-SIMILE bugs)
- sortedcollections
- numpy
- scipy

## Usage

```python
import simile as sml

# Generate pair-specific substitution matrix
S = sml.substitution_matrix(mzs1, mzs2, tolerance=.01)

# Align and score using upper-right quadrant of substitution matrix
score, alignment = sml.pairwise_align(S[:len(mzs1),len(mzs1):])

# Calculate significance of the alignment
pval = sml.alignment_test(S, mzs1, mzs2)

```

## Contributing
Pull requests are welcome.

For major changes, please open an issue first to discuss what you would like to change.

## License
[Modified BSD](https://github.com/biorack/simile/blob/main/license.txt)

## Acknowledgements
The development of SIMILE was made possible by:
* [The U.S. Department of Energy Biological and Environmental Research Program](https://science.energy.gov/ber/)
* [Lawrence Berkeley National Laboratory](http://www.lbl.gov/)
* [The Joint Genome Institute](https://jgi.doe.gov/)
* [The National Energy Research Scientific Computing Center](http://www.nersc.gov/)
