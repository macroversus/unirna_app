# Uni-RNA App

## Introduction
Uni-RNA is a context-aware pre-trained model developed using 1 billion high-quality RNA sequences spanning diverse types, lengths and species. Uni-RNA achieved superior performances in a spectrum of supervised tasks, including structural prediction, m6A methylation and lncRNA localization, as well as unsupervised evolutionary analyses. For therapeutics, Uni-RNA enables closed-loop optimization of 5′UTRs and suppressor tRNAs, improving translation efficiency and stop codon readthrough. Integrated with a pre-trained chemical model, the multimodal framework achieves >35% hit rates in RNA targeting small molecule screening, in contrast to <1% in conventional high throughput screens. Overall, Uni-RNA represents a new research paradigm to dissect and engineer RNA molecules, unlocking the potential of deep learning to expedite RNA research and therapeutics.

## Weights

You can download the pre-trained weights and downstream weights using either of the following methods:

> Note: Password authentication is required

```shell
# pre-trained weights
wget -e robots=off -r -np -nv -nH -R "index.html*" -R "robots.txt" --cut-dirs=1 --compression=gzip -P checkpoints --user=admin --ask-password http://ohgy1146835.bohrium.tech:50004/rna_app/pretrained/

# downstream task weights
wget -e robots=off -r -np -nv -nH -R "index.html*" -R "robots.txt" --cut-dirs=1 --compression=gzip -P checkpoints --user=admin --ask-password http://ohgy1146835.bohrium.tech:50004/rna_app/lite_ckpts/
```


## Description

### RNA sequence feature extraction tool

Use Uni-RNA's pre-trained weights to efficiently characterize sequences. Input a nucleic acid sequence and obtain the corresponding feature vector, which can be used for a series of downstream analyses.

### RNA second structure prediction tool

Use the fine-tuned Uni-RNA to accurately predict the secondary structure of RNA sequences. We provide two fine-tuned versions to choose from, one based on the Uni-RNA self-built dataset and the other based on the RNAStralign dataset.

### mRNA 5'UTR mean ribosome load (MRL) prediction tool

Use the fine-tuned Uni-RNA to accurately predict the mean ribosome load of the 5'UTR region of the mRNA sequence.

### mRNA 3'UTR alternative polyadenylation prediction tool

Use the fine-tuned Uni-RNA to accurately predict alternative polyadenylation events in the 3'UTR region of mRNA sequences.

### RNA m6A modification site prediction tool

Use the fine-tuned Uni-RNA to accurately predict the modification probability of the adenosine at the central position of the sequence

### Long non-coding RNA subcellular localization prediction tool

Use the fine-tuned Uni-RNA to accurately predict the subcellular localization of long non-coding RNA.

### RNA sequence optimization tool

Use the fine-tuned Uni-RNA model to optimize template sequence based on a given biological metric, generating new sequences with enhanced biological activity.

### Affinity screening tool for RNA-targeting small molecules

Given the sequence of the RNA target and the small molecule library to be screened, use the fine-tuned Uni-RNA mode to screen out the small molecule with the strongest binding ability to RNA.