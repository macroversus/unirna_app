Embedding
RNA sequence feature extraction tool
Use Uni-RNA's pre-trained weights to efficiently characterize sequences. Input a nucleic acid sequence and obtain the corresponding feature vector, which can be used for a series of downstream analyses.

二级结构
RNA second structure prediction tool
Use the fine-tuned Uni-RNA to accurately predict the secondary structure of RNA sequences. We provide two fine-tuned versions to choose from, one based on the Uni-RNA self-built dataset and the other based on the RNAStralign dataset.

5UTR
mRNA 5'UTR mean ribosome load (MRL) prediction tool
Use the fine-tuned Uni-RNA to accurately predict the mean ribosome load of the 5'UTR region of the mRNA sequence.

3UTR
mRNA 3'UTR alternative polyadenylation prediction tool
Use the fine-tuned Uni-RNA to accurately predict alternative polyadenylation events in the 3'UTR region of mRNA sequences.

m6A
RNA m6A modification site prediction tool
Use the fine-tuned Uni-RNA to accurately predict the modification probability of the adenosine at the central position of the sequence

lncRNA
Long non-coding RNA subcellular localization prediction tool
Use the fine-tuned Uni-RNA to accurately predict the subcellular localization of long non-coding RNA.

Sequence optimization
RNA sequence optimization tool
Use the fine-tuned Uni-RNA model to optimize template sequence based on a given biological metric, generating new sequences with enhanced biological activity.

RNA-targeted small molecule filtering
Affinity screening tool for RNA-targeting small molecules
Given the sequence of the RNA target and the small molecule library to be screened, use the fine-tuned Uni-RNA mode to screen out the small molecule with the strongest binding ability to RNA.




cd /home/guolvjun/projects/rna_app/rna_app/dash
uv run gunicorn -b 0.0.0.0:50002 -w 4 main_page:server
