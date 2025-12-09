# PRISM-G: Privacy Risk Integrated Score for Multi-representation Genomes

This repository accompanies the PRISM-G pre-print, **“PRISM-G: an interpretable privacy scoring method for assessing risk in synthetic human genome data”** (https://www.biorxiv.org/content/10.1101/2025.10.17.682995). It includes the Python package implementing PRISM-G, along with the datasets and reproducible procedures used in the manuscript.

## Overview

PRISM-G is a privacy scoring framework designed to quantify identity- and attribute-disclosure risk in synthetic human genome data. It evaluates privacy across multiple genetic representations to capture complementary forms of leakage:

- **PLI (Proximity leakage index):** Measures whether synthetic genomes reveal population-level structure patterns by comparing distances between real and synthetic genomes in genetic-coordinate space (PCA).

- **KRI (Kinship replay index):** Detects identity leakage by assessing whether synthetic genomes recreates family structure or long-range dependence. It integrates replay detection, kinship tail inflation, haplotype collison rate, and spectral anomalies observed in genetic relationship matrices. 

- **TLI (Trait-linked leakage):** Estimates exposure through genetic signatures from synthetic sequences using membership inference prediction (MIA) and preservation of rare genetic variants. 

Each component is scaled to a 0–100 score using calibrated baselines (safe vs. leaky generative models), then combined into an overall privacy score. This multi-view approach makes PRISM-G interpretable, robust, and applicable across diverse synthetic genome generation methods.

## Installation

PRISM-G requires the following dependencies:

1. numpy >= 2.3.0
2. pandas >= 2.3.2
3. scikit-learn >= 1.7.2
4. scipy >= 1.16.1
5. matplotlib >= 3.10.6

You can install PRISM-G by using pip with the following command in the source directory that contains the pyproject.toml file:

    pip install .

## Usage

We provide a tutorial notebook demonstrating how to use the PRISM-G package to compute privacy scores, using sample genomes from the 1000 Genomes Project. For synthetic datasets, the tutorial includes examples generated with deep learning models (GAN and RBM) from Yelmen et al. (https://gitlab.inria.fr/ml_genetics/public/artificial_genomes) and a logic-based SAT solver (Genomator) from Burgess et al. Additionally, we include code to generate both *safe* synthetic data using a binomial sampler and *leaky* synthetic data using a copycat generator, enabling users to reproduce the baselines used in the manuscript.

## References

- Yelmen et al. (2021) Creating artificial human genomes using generative neural networks. _PLOS Genetics_. (https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1009303)
- Yelmen et al. (2023) Deep convolutional and conditional neural networks for large-scale genomic data generation. _PLOS Computational Biology_. (https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011584)
- Burgess et al. (2025) Privacy-hardened and hallucination-resistant synthetic data generation with logic-solvers. _Oxford Bioinformatics_. (https://academic.oup.com/bioinformatics/advance-article/doi/10.1093/bioinformatics/btaf600/8314204)
- The 1000 Genomes Project Consortium. (https://www.internationalgenome.org/)







