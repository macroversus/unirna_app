import pandas as pd
from Bio import SeqIO
from deepprotein.inference import LazyInferencer
from .utils import CHECKPOINTS_DIR, PRETRAINED_DIR


def utr_inferencer(input_data: str, output_dir: str) -> int:
    infer = LazyInferencer(
        checkpoint=f"{CHECKPOINTS_DIR}/utr.pth",
        batch_size=1,
        sequence_pretrained=f"{PRETRAINED_DIR}/unirna_L16_E1024_DPRNA500M_STEP400K",
    )
    sequence = [
        {"name": i.description, "utr": str(i.seq), "scaled_rl": 0}
        for i in SeqIO.parse(input_data, "fasta")
    ]
    in_data = pd.DataFrame(sequence)
    result_unirna = infer.run(in_data)

    result_label = [item for lst in result_unirna["scaled_rl"] for item in lst]
    in_data["scaled_rl"] = result_label
    in_data.to_csv(f"{output_dir}/result.csv", index=False)

    return 0
