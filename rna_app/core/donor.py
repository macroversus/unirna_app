from Bio.SeqIO.FastaIO import FastaIterator
import pandas as pd
from .utils import deeprna_infer


def infer_donor(in_data: str | FastaIterator | pd.DataFrame, output_dir: str, return_df: bool = False) -> pd.DataFrame | None:
    ret = deeprna_infer(
        in_data=in_data,
        mission="donor",
        pretrained="L16",
        output_path=f"{output_dir}/result.csv",
        seq_col="seq",
        label_col="label",
        level="token",
        return_df=return_df,
    )
    return ret
