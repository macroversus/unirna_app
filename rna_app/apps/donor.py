import os
import pandas as pd
import pickle
from dp.launching.cli import to_runner, default_minimal_exception_handler
from dp.launching.typing import BaseModel, Field, InputFilePath, String, OutputDirectory,DataSet,Model
from deepprotein.runners.inferencer import LazyInferencer

class LazyInferenceOptions(BaseModel):
    # checkpoint : Model = Field(default="r2_095.pth", description="Path to the checkpoint file")
    # data_list: InputFilePath = Field(..., description="Path to the data list file")
    # dataset: DataSet = Field(..., description="Inference Dataset")
    input_data: InputFilePath = Field(default=None, description="Config")
    # output_dir: OutputDirectory = Field(default="/output", description="Output directory")
    # output_file: String = Field(default="result.pkl", description="Path to the result file")

def lazy_inference_runner(opts: LazyInferenceOptions) -> int:
    os.system("mkdir outputs")
    infer = LazyInferencer(
        # opts.checkpoint.get_full_path(),
        checkpoint = "/tmps/unirna_donor/epoch-8.pth",
        batch_size=1,
        sequence_pretrained='/tmps/deepprotein/unirna_L16_E1024_DPRNA500M_STEP400K/'
    )

    sequence = []
    with open(opts.input_data.get_full_path(), "r") as f:
        current_seq = ''
        for line in f:
            if line.startswith('>'):
                if current_seq != '':
                    sequence.append(current_seq)
                current_seq = ''
            else:
                current_seq += line.strip()
        sequence.append(current_seq)
                
    data_list = pd.DataFrame({'seq':sequence,'label':0})
    result_unirna = infer.run(data_list)
    
    result_label = [num for num in result_unirna['label'] ]
    result_df = pd.DataFrame({'seq': sequence, 'label':result_label})
    result_df.to_csv("./outputs/result.csv", index=False)

    return 0

def to_parser():
    return to_runner(
        LazyInferenceOptions,
        lazy_inference_runner,
        version="0.1.0",
        exception_handler=default_minimal_exception_handler,
    )

if __name__ == "__main__":
    import sys
    sys.exit(to_parser()(sys.argv[1:]))