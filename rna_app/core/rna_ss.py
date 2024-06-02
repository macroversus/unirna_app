from Bio.SeqIO.FastaIO import FastaIterator
import pandas as pd
import numpy as np
from .utils import deeprna_infer


def infer_ss(
    in_data: str | FastaIterator | pd.DataFrame,
    output_dir: str,
    model_type: str,
    return_df: bool = False,
    keep_prob: bool = False,
) -> pd.DataFrame | None:
    assert model_type in [
        "unirna",
        "archiveii",
    ], f"model_type {model_type} not supported. Supported model_type: ['unirna', 'archiveii']"
    ret = deeprna_infer(
        in_data=in_data,
        mission="ss_unirna" if model_type == "unirna" else "ss_archiveii",
        pretrained="L16",
        seq_col="seq",
        label_col="label",
        level="token",
        return_df=True,
    )
    ss_texts = []
    for v in ret["label"].values:
        matrix = np.array(v)
        dot_of_interest = np.where(matrix > 0.5)
        connections = pd.DataFrame([*dot_of_interest, matrix[dot_of_interest]]).T
        connections.columns = ["i", "j", "proba"]
        connections.sort_values(by="proba", ascending=False, inplace=True)
        connections[["i", "j"]] = (
            connections[["i", "j"]]
            .apply(
                lambda x: [x["i"], x["j"]] if x["i"] < x["j"] else [x["j"], x["i"]],
                axis=1,
                result_type="expand",
            )
            .astype(int)
        )
        connections.drop_duplicates(subset="i", keep="first", inplace=True)
        ss_text = ["."] * matrix.shape[0]
        for i, j in connections[["i", "j"]].values:
            ss_text[i] = "("
            ss_text[j] = ")"
        ss_text = "".join(ss_text)
        ss_texts.append(ss_text)
    ret["secondary_structure"] = ss_texts
    if not keep_prob:
        ret.drop(columns=["label"], inplace=True)
    ret.to_csv(f"{output_dir}/result.csv", index=False)
    if return_df:
        return ret
