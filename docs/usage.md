# Tutorial: Evaluating Privacy Risk with PRISM-G

This tutorial introduces the basic workflow for using **PRISM-G** to evaluate the privacy risk of synthetic genomic datasets.

Throughout the tutorial, we will analyze synthetic genotype datasets generated from the [1000 Genomes Project](https://www.internationalgenome.org/), a publicly available resource comprising genotype data from approximately 2,500 individuals across 26 populations worldwide. The synthetic datasets were generated using three representative generative approaches:

- Generative Adversarial Network (GAN)
- Restricted Boltzmann Machines (RBM)
- Genomator, a logic-based SAT solver

The GAN and RBM datasets were generated using the implementations described by **Yelmen et al. (2021)**[^yelmen], while the Genomator datasets were generated using the method proposed by **Burgess et al. (2025)**[^burgess].

In this tutorial, you will learn how to:

- Estimate the three PRISM-G privacy components:
    - Proximity Leakage Index (PLI)
    - Kinship Replay Index (KRI)
    - Trait-Linked Leakage Index (TLI)
- Interpret the individual privacy diagnostics produced by each component.
- Aggregate the component scores into a calibrated privacy score.
- Compare and rank different synthetic datasets according to their estimated privacy risk.

Finally, we will illustrate how PRISM-G can be combined with a downstream utility metric to analyze the privacy–utility trade-off. As an example application, we will perform a genetic ancestry prediction task using the population labels from the 1000 Genomes Project, and construct a **privacy–utility Pareto frontier** to compare the different synthetic data generators.

**References**

[^yelmen]: Yelmen, B., Decelle, A., Ongaro, L., *et al.* (2021). *Creating artificial human genomes using generative neural networks.* **PLOS Genetics**, 17(2), e1009303. https://doi.org/10.1371/journal.pgen.1009303

[^burgess]: Burgess, M. A., Hosking, B., Reguant, R., *et al.* (2025). *Privacy-hardened and hallucination-resistant synthetic data generation with logic-solvers.* **Bioinformatics**. https://doi.org/10.1093/bioinformatics/btaf600