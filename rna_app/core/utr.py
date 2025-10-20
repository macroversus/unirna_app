from Bio.SeqIO.FastaIO import FastaIterator
import pandas as pd
from .utils import deeprna_infer


def infer_utr(
    in_data: str | FastaIterator | pd.DataFrame,
    output_dir: str,
    return_df: bool = False,
) -> pd.DataFrame | None:
    ret = deeprna_infer(
        in_data=in_data,
        mission="utr",
        pretrained="L16",
        output_path=f"{output_dir}/result.csv",
        seq_col="utr",
        label_col="scaled_rl",
        level="seq",
        out_seq_colname="utr_sequence",
        return_df=return_df,
    )
    return ret
