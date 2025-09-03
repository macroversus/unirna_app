# Uni-RNA App

## Installation

```shell
pip install --no-cache-dir -U pip setuptools wheel
pip install --no-cache-dir git+ssh://git@git.dp.tech/macromolecule/deepprotein@deeprna
pip install --no-cache-dir git+ssh://git@git.dp.tech/macromolecule/unirna_transformers@main
pip install --no-cache-dir -e .
```


## Weights

You can download the pre-trained weights and downstream weights using either of the following methods:
Note: Password authentication is required
### Option 1: Using wget commands

Download pre-trained weights:
```shell
wget -e robots=off -r -np -nv -nH -R "index.html*" -R "robots.txt" --cut-dirs=1 --compression=gzip -P checkpoints --user=admin --ask-password http://ohgy1146835.bohrium.tech:50004/rna_app/pretrained/
```

Download downstream task weights:
```shell
wget -e robots=off -r -np -nv -nH -R "index.html*" -R "robots.txt" --cut-dirs=1 --compression=gzip -P checkpoints --user=admin --ask-password http://ohgy1146835.bohrium.tech:50004/rna_app/lite_ckpts/
```

### Option 2: Using shell script 

Simply run:
```shell
zsh download_checkpoints.sh
```

> Note: Both methods will download the same files. Choose either one that suits you.
## Description

### RNA Secondary Structure Prediction

We provided two models, trained on the Uni-RNA 50% threshold secondary structure dataset and MXfold2 (RNAStralign) dataset. The test sets were also provided for download.

### 5' UTR Mean Ribsomal Loading Prediction

We provided model trained on Moderna datasets.

### LncRNA Localization

This is a classification model for LncRNA localization.

### Embedding Inference

We provided easily used APIs for Uni-RNA model embedding inference.

### m6A Modification Prediction

We provided a model for m6A modification prediction.