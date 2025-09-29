from Bio.SeqIO.FastaIO import FastaIterator
import pandas as pd
from typing import Union, Optional
from .utils import deeprna_infer


def infer_apa(
    in_data: Union[str, FastaIterator, pd.DataFrame],
    output_dir: str,
    return_df: bool = False,
    verbose: bool = False,
) -> Optional[pd.DataFrame]:
    
    ret = deeprna_infer(
        in_data=in_data,
        mission="apa",
        pretrained="L16",
        output_path=f"{output_dir}/result.csv",
        seq_col="seq",
        label_col="label",
        level="seq",
        out_seq_colname="RNA_utr_sequence",
        out_label_colname="RNA_3apa_score",
        return_df=return_df,
    )
    return ret