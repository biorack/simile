# SIMILE

SIMILE (Significant Interrelation of MS/MS Ions via Laplacian Embedding) is a Python library for interrelating fragmentation spectra with significance estimation and is robust to multiple differences in chemical structure.
[bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2021.02.24.432767v1)

### New in V2:
- Precursor-based neutral loss difference counts can be used in addition to the original MZ difference counts
- Maximum weight matching is used instead of original monotonic alignment method with improved performance
- Gap penalties to further improve significance estimation
- Multiple matching in addition to original pairwise matching for fragment centric analyses
- MUCH faster mass delta counting and significance testing
- Matching ions report summarizing all scores and mass deltas with metadata

![SIMILE Flow](SimileFig1Vert.png "SIMILE")

## Installation

Use the package manager [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/index.html) to install environment-base.yml for minimum requirements. Alternatively, use environment.yml to run the example notebook.

```bash
conda env create -f environment-base.yml
```

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/biorack/simile/HEAD)

## Python dependencies
- python3 (pinned to 3.7 currently due to non-SIMILE bugs)
- numpy
- scipy
- pandas

## Usage

```python
import simile as sml

# Generate fragmentation similarity matrix
S, spec_ids = sml.similarity_matrix(mzs, pmzs=pmzs, tolerance=.01)

# Generate pro/con comparison matrix such that 
# interspectral comparisons are 1 (pro) and
# intraspectral comparisions are -1 (con)
C = sml.inter_intra_compare(spec_ids)

# Generate max weight matching for similarity matrix
M = sml.pairwise_match(S)

# Score fragment ion similarity using previous matrices
score, scores, probs = sml.match_scores(S, C, M, spec_ids, gap_penalty=4)

# Calculate significance of max weight matching between fragment ions
pval, null_dist = sml.mcp_test(scores, probs, return_dist=True, log_size=4)

# Report back mass deltas and scores for simile comparison
df = sml.matching_ions_report(S, C, M, mzs, pmzs)

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
