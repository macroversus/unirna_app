from .utils import deeprna_infer


def infer_m6a(in_filepath: str, output_dir: str) -> int:
    deeprna_infer(
        in_filepath=in_filepath,
        mission="m6a",
        pretrained="L16",
        output_path=f"{output_dir}/result.csv",
        seq_col="seq",
        label_col="label",
        level="token",
    )
    return 0
