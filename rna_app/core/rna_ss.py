from Bio.SeqIO.FastaIO import FastaIterator
import pandas as pd
from .utils import deeprna_infer


def infer_ss(in_data: str | FastaIterator | pd.DataFrame, output_dir: str, model_type: str, return_df: bool = False) -> pd.DataFrame | None:
    assert model_type in [
        "unirna",
        "archiveii",
    ], f"model_type {model_type} not supported. Supported model_type: ['unirna', 'archiveii']"
    ret = deeprna_infer(
        in_data=in_data,
        mission="ss_unirna" if model_type == "unirna" else "ss_archiveii",
        pretrained="L16",
        output_path=f"{output_dir}/result.csv",
        seq_col="seq",
        label_col="label",
        level="token",
        return_df=return_df,
    )
    return ret
