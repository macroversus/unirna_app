# Uni-RNA App

❗️**Note**: This project is still in development and is not yet ready for use.

## Installation

```shell
pip install --no-cache-dir -U pip setuptools wheel
pip install --no-cache-dir -e .
zsh download_checkpoints.sh
```

## Description

### RNA Secondary Structure Prediction

We provided two models, trained on the Uni-RNA 50% threshold secondary structure dataset and MXfold2 (RNAStralign) dataset. The test sets were also provided for download.

### 5' UTR Mean Ribsomal Loading Prediction

We provided model trained on Moderna datasets.

### LncRNA Localization

This is a classification model for LncRNA localization.

### piRNA Prediction

This is a fine-tuned model for piRNA determination.

### Splice Site Prediction

We provided two fine-tuned models for donor and acceptor site prediction.

This app can choose between Donor and Acceptor versions.

### Embedding Inference

We provided easily used APIs for Uni-RNA model embedding inference.

### m6A Modification Prediction

We provided a model for m6A modification prediction.