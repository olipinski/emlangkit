[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Emergent Language Analysis Toolkit

This toolkit aims to collect all metrics currently used in emergent
communication research into one place. The usage should be convenient and the
inputs should be standardised, to ease adoption and spread of these metrics.

## Installation

To install emlangkit, run `pip install emlangkit`.

Automatic tests are run for Python 3.9, 3.10, 3.11, 3.12.

## Usage

All metrics are available through the `Language` class in `emlangkit.Language`.
This class accepts two numpy arrays as inputs - messages and observations. These
are then used, with some possible speedups, to calculate any requested metric,
as per example below

```python
import numpy as np
from emlangkit import Language

messages = np.array(
    [[1, 2, 0, 3, 4], [1, 2, 2, 3, 4], [1, 2, 2, 3, 0], [1, 0, 0, 1, 2]]
)
observations = np.array([[4, 4], [4, 3], [3, 2], [1, 4]])

lang = Language(messages=messages, observations=observations)

score, p_value = lang.topsim()

# Mutual information already requires both language and observation entropy
mi = lang.mutual_information()

# So this call uses less computation
lang_entropy = lang.language_entropy()
```

## Metrics

Currently available metrics, with their implementations as per below.

- Entropy \[1,2\] (`emlangkit.metrics.entropy`)
- Mutual Information \[1,2\] (`emlangkit.metrics.mutual_information`)
- Topographic Similarity \[1,2,3\] (`emlangkit.metrics.topsim`)
- Positional Disentanglement \[2,4\] (`emlangkit.metrics.posdis`)
- Bag-of-Words Disentanglement \[2,4\] (`emlangkit.metrics.bosdis`)
- M_previous^n \[5\] (`emlangkit.metrics.mpn`)
- Harris' Articulation Scheme \[6\] (`emlangkit.metrics.has`)
- Non-Compositional Normalised Pointwise Mutual Information (NPMI) \[7\]
  (`emlangkit.metrics.nc_npmi`)

## Contributing

All pull requests are welcome! Just please make sure to install pre-commit and
run the pytests before submitting a PR. Additionally, if a lot of new code is
added, please also add the relevant tests.

## Related Libraries

This is a non-exhaustive list of libraries related to EC research. Please feel
free to open a PR to add to it!

- EGG - https://github.com/facebookresearch/EGG
- Harris' Articulation Scheme - https://github.com/wedddy0707/HarrisSegmentation
- CGI -
  https://github.com/wedddy0707/categorial_grammar_induction_of_emergent_language

## Citation

If you find emlangkit useful in your work, please cite it as below:

```
@software{lipinski_emlangkit_2023,
        title = {emlangkit: Emergent Language Analysis Toolkit},
        url = {https://github.com/olipinski/emlangkit},
        author = {Lipinski, Olaf},
        year = {2023}
}
```

## Sources

Most of the base metrics are inspired or taken from either
[EGG](https://github.com/facebookresearch/EGG), or code from the paper
"Catalytic Role Of Noise And Necessity Of Inductive Biases In The Emergence Of
Compositional Communication"
[here](https://proceedings.neurips.cc/paper/2021/hash/c2839bed26321da8b466c80a032e4714-Abstract.html).

Citations for the metrics and parts of this software:

- \[1\] L. Kucinski, T. Korbak, P. Kolodziej, and P. Milos, ‘Catalytic Role Of
  Noise And Necessity Of Inductive Biases In The Emergence Of Compositional
  Communication’, NeurIPS 2021
- \[2\] E. Kharitonov, R. Chaabouni, D. Bouchacourt, and M. Baroni, ‘EGG: a
  toolkit for research on Emergence of lanGuage in Games’, EMNLP-IJCNLP 2019
- \[3\] H. Brighton and S. Kirby, ‘Understanding Linguistic Evolution by
  Visualizing the Emergence of Topographic Mappings’, Artificial Life, vol. 12,
  no. 2, pp. 229–242, Apr. 2006
- \[4\] R. Chaabouni, E. Kharitonov, D. Bouchacourt, E. Dupoux, and M. Baroni,
  ‘Compositionality and Generalization In Emergent Languages’, ACL 2020
- \[5\] O. Lipinski, A. J. Sobey, F. Cerutti, and T. J. Norman, ‘On Temporal
  References in Emergent Communication’. arXiv.2310.06555
- \[6\] R. Ueda, T. Ishii, and Y. Miyao, ‘On the Word Boundaries of Emergent
  Languages Based on Harris’s Articulation Scheme’, ICLR 2023
- \[7\] O. Lipinski, A. J. Sobey, F. Cerutti, and T. J. Norman, ‘Speaking Your
  Language: Spatial Relationships in Interpretable Emergent Communication’,
  NeurIPS 2024
