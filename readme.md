# Uni-RNA App

## Weights

You can download the pre-trained weights and downstream weights using either of the following methods:

> Note: Password authentication is required

### Option 1: Using wget commands

```shell
# pre-trained weights
wget -e robots=off -r -np -nv -nH -R "index.html*" -R "robots.txt" --cut-dirs=1 --compression=gzip -P checkpoints --user=admin --ask-password http://ohgy1146835.bohrium.tech:50004/rna_app/pretrained/

# downstream task weights
wget -e robots=off -r -np -nv -nH -R "index.html*" -R "robots.txt" --cut-dirs=1 --compression=gzip -P checkpoints --user=admin --ask-password http://ohgy1146835.bohrium.tech:50004/rna_app/lite_ckpts/
```

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