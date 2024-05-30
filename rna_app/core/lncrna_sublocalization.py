from .utils import deeprna_infer


def infer_lncrna_sublocalization(in_filepath: str, output_dir: str) -> int:
    deeprna_infer(
        in_filepath=in_filepath,
        mission="lncrna_sublocalization",
        pretrained="L16",
        output_path=f"{output_dir}/result.csv",
        seq_col="cdna",
        label_col="tag",
        level="token",
        out_seq_colname="lncrna_sequence",
    )
    return 0
