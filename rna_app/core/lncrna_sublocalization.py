from Bio.SeqIO.FastaIO import FastaIterator
import pandas as pd
from .utils import deeprna_infer


def infer_lncrna_sublocalization(in_data: str | FastaIterator | pd.DataFrame, output_dir: str, return_df: bool = False) -> pd.DataFrame | None:
    ret = deeprna_infer(
        in_data=in_data,
        mission="lncrna_sublocalization",
        pretrained="L16",
        output_path=f"{output_dir}/result.csv",
        seq_col="cdna",
        label_col="tag",
        level="token",
        out_seq_colname="lncrna_sequence",
        return_df=return_df,
    )
    return ret
