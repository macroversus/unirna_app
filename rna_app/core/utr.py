from .utils import deeprna_infer


def infer_utr(in_filepath: str, output_dir: str) -> int:
    deeprna_infer(
        in_filepath=in_filepath,
        mission="utr",
        pretrained="L16",
        output_path=f"{output_dir}/result.csv",
        seq_col="utr",
        label_col="scaled_rl",
        level="seq",
        out_seq_colname="utr_sequence",
    )
    return 0
