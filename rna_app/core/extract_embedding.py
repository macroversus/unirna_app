from Bio import SeqIO
from Bio.SeqIO.FastaIO import FastaIterator
import pandas as pd
from tqdm import tqdm
import pickle
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from .utils import PRETRAINED


def extract_embedding(
    in_data: str | FastaIterator | pd.DataFrame,
    output_dir: str,
    pretrained: str,
    output_attentions: bool,
) -> int:
    assert (
        pretrained in PRETRAINED.keys()
    ), f"pretrained {pretrained} not supported. Supported pretrained: {list(PRETRAINED.keys())}"
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED[pretrained])
    model = AutoModel.from_pretrained(PRETRAINED[pretrained])
    sequences = [str(i.seq) for i in SeqIO.parse(in_data, "fasta")]
    outputs = {}
    for seq in tqdm(sequences):
        token = tokenizer(seq, return_tensors="pt")
        output = model(**token, output_attentions=output_attentions)
        outputs[seq] = output
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{output_dir}/result.pickle", "wb") as f:
        pickle.dump(outputs, f)

    return 0
