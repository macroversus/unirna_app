import os
import pandas as pd
import pickle
from dp.launching.cli import to_runner, default_minimal_exception_handler
from dp.launching.typing import BaseModel, Field, InputFilePath, String, Enum
from deepprotein.runners.inferencer import LazyInferencer

class ModelTypeOptions(String, Enum):
    """
    二选一
    """
    unirna  = "Type_1 - trained on the Uni-RNA 50% threshold secondary structure dataset"
    archiveii  = "Type_2 - trained on the MXfold2 (RNAStralign) dataset"



class LazyInferenceOptions(BaseModel):

    input_data: InputFilePath = Field(default=None, description="Upload fasta file")
    model_type: ModelTypeOptions = Field(default="Type_1 - trained on the Uni-RNA 50% threshold secondary structure dataset", description="Model type")
    # output_dir: OutputDirectory = Field(default="/output", description="Output directory")
    # output_file: String = Field(default="result.pkl", description="Path to the result file")


def lazy_inference_runner(opts: LazyInferenceOptions) -> int:
    os.system("mkdir outputs")
    # pdb.set_trace()
    # if opts.model_type == "unirna":
    #     infer = LazyInferencer(
    #         # opts.checkpoint.get_full_path(),
    #         checkpoint = "/root/unirna_ss/epoch-11.pth",
    #         batch_size=1,
    #         sequence_pretrained='/root/deepprotein/unirna_L16_E1024_DPRNA500M_STEP400K/'
    #     )
        
    # elif opts.model_type == "archiveii":
    #     infer = LazyInferencer(
    #         # opts.checkpoint.get_full_path(),
    #         checkpoint = "/root/unirna_ss/epoch-19.pth",
    #         batch_size=1,
    #         sequence_pretrained='/root/deepprotein/archiveii_L16_E1024_DPRNA500M_STEP400K/'
    #     )
    # else:
    #     raise ValueError(f"Invalid model_type: {opts.model_type}")
    
    if opts.model_type == ModelTypeOptions.unirna.value:
        infer = LazyInferencer(
            checkpoint = "/tmps/unirna_ss/epoch-11.pth",
            batch_size=1,
            sequence_pretrained='/tmps/deepprotein/unirna_L16_E1024_DPRNA500M_STEP400K/'
        )
    elif opts.model_type == ModelTypeOptions.archiveii.value:
        infer = LazyInferencer(
            checkpoint = "/tmps/unirna_ss/epoch-19.pth",
            batch_size=1,
            sequence_pretrained='/tmps/deepprotein/unirna_L16_E1024_DPRNA500M_STEP400K/'
        )
    else:
        raise ValueError(f"Invalid model_type: {opts.model_type}")


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