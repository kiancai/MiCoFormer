<p align="center">
  <img src="assets/micoformer-logo.png" alt="MiCoFormer" width="680">
</p>

<p align="center">
  <a href="https://github.com/kiancai/MiCoFormer/tree/v0.5"><img src="https://img.shields.io/badge/Code-v0.5-355C7D?logo=github" alt="Code v0.5 on GitHub"></a>
  <a href="https://huggingface.co/kiancai/MiCoFormer"><img src="https://img.shields.io/badge/Model-Epoch%2096-D6A338?logo=huggingface" alt="MiCoFormer epoch 96 on Hugging Face"></a>
  <a href="https://huggingface.co/datasets/kiancai/MCFCorpus"><img src="https://img.shields.io/badge/MCFCorpus-V20261119-508A7C?logo=huggingface" alt="MCFCorpus V20261119 on Hugging Face"></a>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/Code%20license-MIT-64748B" alt="Code license MIT"></a>
  <a href="https://github.com/kiancai/MiCoFormer/blob/v0.5/pyproject.toml"><img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&amp;logoColor=white" alt="Python 3.11 or newer"></a>
  <a href="https://github.com/kiancai/MiCoFormer/blob/v0.5/requirements.txt"><img src="https://img.shields.io/badge/PyTorch-2.6.0-EE4C2C?logo=pytorch&amp;logoColor=white" alt="PyTorch 2.6.0"></a>
</p>

## MCFCorpus

We integrated data from 10 databases, including MicrobeAtlas, Qiita, MicroBioMap, PRIME, and ResMicroDb, and reannotated sequence data against Greengenes2. The AnnData object contains genus-level profiles for 3,334,541 samples across 8,114 genera.

[![MCFCorpus data integration](assets/mcfcorpus-integration.png)](assets/mcfcorpus-integration.pdf)

## Model architecture

Each genus prior combines a phylogenetic embedding from the reference tree with a protein-derived embedding generated using ESM-2 and BacFormer and aggregated across genomes.

MiCoFormer embeds rclr-transformed abundances and adds them to genus priors for Transformer encoding. Pretraining combines masked abundance reconstruction, held-out genus presence and abundance prediction, and UniFrac-based neighborhood matching.

[![MiCoFormer model architecture](assets/micoformer-architecture.png)](assets/micoformer-architecture.pdf)
