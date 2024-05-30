from .utils import deeprna_infer


def infer_ss(in_filepath: str, output_dir: str, model_type: str) -> int:
    deeprna_infer(
        in_filepath=in_filepath,
        mission="ss_unirna" if model_type == "unirna" else "ss_archiveii",
        pretrained="L16",
        output_path=f"{output_dir}/result.csv",
        seq_col="seq", 
        label_col="label",
        level="token",
    )
    return 0
