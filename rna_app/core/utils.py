from pathlib import Path
from Bio import SeqIO
import pandas as pd
from typing import Optional
from deepprotein.inference import LazyInferencer

ROOT_DIR = Path(__file__).parent.parent.parent
PRETRAINED_DIR = ROOT_DIR / "checkpoints/pretrained"
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints/lite_ckpts"

CHEKPOINTS = {
    "utr": CHECKPOINTS_DIR / "utr.pth",
    "ss_unirna": CHECKPOINTS_DIR / "unirna_ss_0.5_thrs_dataset.pth",
    "ss_archiveii": CHECKPOINTS_DIR / "unirna_ss_archiveii_dataset.pth",
    "acceptor": CHECKPOINTS_DIR / "acceptor.pth",
    "donor": CHECKPOINTS_DIR / "donor.pth",
    "lncrna_sublocalization": CHECKPOINTS_DIR / "lncrna_sublocalization.pth",
    "m6a": CHECKPOINTS_DIR / "m6a.pth",
    "pirna": CHECKPOINTS_DIR / "pirna.pth",
}

PRETRAINED = {
    "L8": PRETRAINED_DIR / "unirna_L8_E512_STEP290K_DPRNA100M",
    "L12": PRETRAINED_DIR / "unirna_L12_E768_STEP210K_DPRNA100M",
    "L16": PRETRAINED_DIR / "unirna_L16_E1024_DPRNA500M_STEP400K",
    "L24": PRETRAINED_DIR / "unirna_L24_E1280_STEP180K_DPRNA500M",
}

def read_in_data(in_filepath: str, seq_col: str = "seq", label_col: str = "label") -> pd.DataFrame:
    """_summary_

    Args:
        in_filepath (str): 输入文件路径
        seq_col (str, optional): RNA序列所在列名. Defaults to "seq".
        label_col (str, optional): 训练时label所在列名. Defaults to "label".

    Raises:
        ValueError: 输入文件格式不支持

    Returns:
        pd.DataFrame: 可用于deeprna推理的DataFrame
    """    
    for i in ["name", "description", "id"]:
        if i not in [seq_col, label_col]:
            name_colname = i
            break
    if in_filepath.endswith(("fasta", "fa", "fna")):
        out = pd.DataFrame(
            [
                {name_colname: i.description, seq_col: str(i.seq), label_col: 0}
                for i in SeqIO.parse(in_filepath, "fasta")
            ]
        )
    elif in_filepath.endswith("csv"):
        out = pd.read_csv(in_filepath)
    elif in_filepath.endswith("tsv"):
        out = pd.read_csv(in_filepath, sep="\t")
    elif in_filepath.endswith("xlsx"):
        out = pd.read_excel(in_filepath)
    elif in_filepath.endswith("pkl"):
        out = pd.read_pickle(in_filepath)
    else:
        raise ValueError("Input file format not supported")
    return out

def save_dataframe(df: pd.DataFrame, output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if output_path.endswith("csv"):
        df.to_csv(output_path, index=False)
    elif output_path.endswith("tsv"):
        df.to_csv(output_path, sep="\t", index=False)
    elif output_path.endswith("xlsx"):
        df.to_excel(output_path, index=False)
    elif output_path.endswith("pkl"):
        df.to_pickle(output_path)
    else:
        raise ValueError("Output file format not supported")

def deeprna_infer(in_filepath: str, mission: str, pretrained: str, output_path: Optional[str] = None, return_df: bool = False, seq_col: str = "seq", label_col: str = "label", level: str = "seq"):
    assert mission in CHEKPOINTS.keys(), f"mission {mission} not supported. Supported missions: {list(CHEKPOINTS.keys())}"
    assert pretrained in PRETRAINED.keys(), f"pretrained {pretrained} not supported. Supported pretrained: {list(PRETRAINED.keys())}"
    infer = LazyInferencer(
        checkpoint=CHEKPOINTS[mission],
        batch_size=1,
        sequence_pretrained=PRETRAINED[pretrained],
    )
    in_data = read_in_data(in_filepath = in_filepath, seq_col = seq_col, label_col = label_col)
    result_unirna = infer.run(in_data)
    if level == "seq":
        in_data[label_col] = [item for lst in result_unirna[label_col] for item in lst]
    elif level == "token":
        in_data[label_col] = [lst for lst in result_unirna[label_col]]
    else:
        raise ValueError("level should be 'seq' or 'token'")
    if output_path:
        save_dataframe(in_data, output_path)
    if return_df:
        return in_data