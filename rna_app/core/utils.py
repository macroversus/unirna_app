from pathlib import Path
import torch
from Bio import SeqIO
from Bio.SeqIO.FastaIO import FastaIterator
import pandas as pd
from typing import Optional
import unirna_tf
from deepprotein.inference import LazyInferencer
import diskcache

ROOT_DIR = Path(__file__).parent.parent.parent
PRETRAINED_DIR = ROOT_DIR / "checkpoints/pretrained"
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints/lite_ckpts"
CACHE_DIR = ROOT_DIR / ".cache"

CHEKPOINTS = {
    "apa" : CHECKPOINTS_DIR / "apa.pth",
    "utr": CHECKPOINTS_DIR / "utr.pth",
    "ss_unirna": CHECKPOINTS_DIR / "unirna_ss_0.5_thrs_dataset.pth",
    "ss_archiveii": CHECKPOINTS_DIR / "unirna_ss_archiveii_dataset.pth",
    "acceptor": CHECKPOINTS_DIR / "acceptor.pth",
    "donor": CHECKPOINTS_DIR / "donor.pth",
    "lncrna_sublocalization": CHECKPOINTS_DIR / "lncrna_sublocalization.pth",
    "m6a": CHECKPOINTS_DIR / "m6a.pth",
    "pirna": CHECKPOINTS_DIR / "pirna.pth",
    "trna_seq_optimization": CHECKPOINTS_DIR / "best_unirna_reg_model.pth",
    "5utr_seq_optimization": CHECKPOINTS_DIR / "5utr_best_unirna_reg_model.pth",
}

PRETRAINED = {
    "L8": PRETRAINED_DIR / "unirna_L8_E512_STEP290K_DPRNA100M",
    "L12": PRETRAINED_DIR / "unirna_L12_E768_STEP210K_DPRNA100M",
    "L16": PRETRAINED_DIR / "unirna_L16_E1024_DPRNA500M_STEP400K",
    "L24": PRETRAINED_DIR / "unirna_L24_E1280_STEP180K_DPRNA500M",
}


def get_cache(cache_name: str, size_limit: int = 2 * 1024**3) -> diskcache.Cache:
    """
    Get a diskcache.Cache instance with specified name and size limit.

    Args:
        cache_name: Name of the cache (will be used as subdirectory name)
        size_limit: Maximum cache size in bytes (default: 2GB)

    Returns:
        diskcache.Cache instance
    """
    cache_path = CACHE_DIR / cache_name
    cache_path.mkdir(parents=True, exist_ok=True)
    return diskcache.Cache(str(cache_path), size_limit=size_limit)


def read_in_data(
    in_data: str | FastaIterator | pd.DataFrame,
    seq_col: str = "seq",
    label_col: str = "label",
) -> pd.DataFrame:
    """_summary_

    Args:
        in_data (str | FastaIterator | pd.DataFrame): 输入数据，可以是文件路径，FastaIterator或者pd.DataFrame
        seq_col (str, optional): RNA序列所在列名. Defaults to "seq". 不可用"name"
        label_col (str, optional): 训练时label所在列名. Defaults to "label".

    Raises:
        ValueError: 输入文件格式不支持

    Returns:
        pd.DataFrame: 可用于deeprna推理的DataFrame
    """
    if isinstance(in_data, str):
        if in_data.endswith(("fasta", "fa", "fna")):
            out = pd.DataFrame(
                [
                    {"name": i.description, seq_col: str(i.seq), label_col: 0}
                    for i in SeqIO.parse(in_data, "fasta")
                ]
            )
        elif in_data.endswith("csv"):
            out = pd.read_csv(in_data)
        elif in_data.endswith("tsv"):
            out = pd.read_csv(in_data, sep="\t")
        elif in_data.endswith("xlsx"):
            out = pd.read_excel(in_data)
        else:
            raise ValueError("Input file format not supported")
    elif isinstance(in_data, FastaIterator):
        out = pd.DataFrame(
            [
                {"name": i.description, seq_col: str(i.seq), label_col: 0}
                for i in in_data
            ]
        )
    else:
        out = in_data
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


def deeprna_infer(
    in_data: str | FastaIterator | pd.DataFrame,
    mission: str,
    pretrained: str,
    output_path: Optional[str | Path] = None,
    return_df: bool = False,
    seq_col: str = "seq",
    label_col: str = "label",
    level: str = "seq",
    out_seq_colname: Optional[str] = None,
    out_label_colname: Optional[str] = None,
    **kwargs,
):
    assert (
        mission in CHEKPOINTS.keys()
    ), f"mission {mission} not supported. Supported missions: {list(CHEKPOINTS.keys())}"
    assert (
        pretrained in PRETRAINED.keys()
    ), f"pretrained {pretrained} not supported. Supported pretrained: {list(PRETRAINED.keys())}"
    infer = LazyInferencer(
        checkpoint=CHEKPOINTS[mission],
        batch_size=1,
        sequence_pretrained=PRETRAINED[pretrained],
    )
    in_data = read_in_data(in_data=in_data, seq_col=seq_col, label_col=label_col)
    result_unirna = infer.run(in_data)
    if level == "seq":
        in_data[label_col] = [item for lst in result_unirna[label_col] for item in lst]
    elif level == "token":
        in_data[label_col] = [lst for lst in result_unirna[label_col]]
    else:
        raise ValueError("level should be 'seq' or 'token'")
    torch.cuda.empty_cache()
    if out_seq_colname:
        in_data.rename(columns={seq_col: out_seq_colname}, inplace=True)
    if out_label_colname:
        in_data.rename(columns={label_col: out_label_colname}, inplace=True)
    if output_path:
        save_dataframe(in_data, output_path)
    if return_df:
        return in_data
