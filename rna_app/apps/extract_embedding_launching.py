import os
import pickle
from dp.launching.cli import to_runner, default_minimal_exception_handler
from dp.launching.typing import BaseModel,Optional,Field, InputFilePath, String, Enum,validator
import unirna_tf
from transformers import AutoTokenizer, AutoModel

class ModelTypeOptions(String, Enum):

    unirna_L8_E512_STEP290K_DPRNA100M = "unirna_L8_E512_STEP290K_DPRNA100M"
    unirna_L12_E768_STEP210K_DPRNA100M = "unirna_L12_E768_STEP210K_DPRNA100M"
    unirna_L16_E1024_DPRNA500M_STEP400K = "unirna_L16_E1024_DPRNA500M_STEP400K"
    unirna_L24_E1280_STEP180K_DPRNA500M = "unirna_L24_E1280_STEP180K_DPRNA500M"

# class AdvancedSettings(BaseModel):
#     """
#     1. 选择unirna 模型
#     2. 选择是否output attentions
#     """
#     model_type: ModelTypeOptions = Field(default="unirna_L16_E1024_DPRNA500M_STEP400K", title= "Model Type",description="Choose which unirna weights to use. The default is 'unirna_L16_E1024_DPRNA500M_STEP400K'.")
#     output_attentions: Optional[bool] = Field(default=True, title = 'Output Attentions',description="Decide whether to output attentions or not. If set to True, the model will output attention weights. The default is True.")


class SingleOptions(BaseModel):
    """
    input file set
    """
    input_data: InputFilePath = Field(default=None, description="Upload fasta file")
    model_type: ModelTypeOptions = Field(default="unirna_L16_E1024_DPRNA500M_STEP400K", title= "Model Type",description="Choose which unirna weights to use. The default is 'unirna_L16_E1024_DPRNA500M_STEP400K'.")
    output_attentions: Optional[bool] = Field(default=True, title = 'Output Attentions',description="Decide whether to output attentions or not. If set to True, the model will output attention weights. The default is True.")
    # AdvancedSettings: AdvancedSettings = Field(default=AdvancedSettings(), title="Advanced Settings", description="Advanced settings for the model.")


    # @validator("input_fasta_file")
    # def check_input_s(cls, v: InputFilePath):
    #     from pathlib import Path
    #     path = Path(v.get_full_path())
    #     if not path.exists():
    #         raise ValueError(f"文件({v})不存在")
    #     if not path.suffix.lower() == '.fasta' or path.suffix.lower() == '.fa':
    #         raise ValueError(f"文件({v})不是有效的fasta文件")
    #     return v
    
    # AdvancedSettings: AdvancedSettings = Field(..., title="Advanced Settings", description="Advanced settings for the model.")



def extract_embeding_runner(opts:SingleOptions) -> int:
    if os.path.exists("./outputs") is False:
        os.system("mkdir outputs")  
        

    if opts.model_type == 'unirna_L8_E512_STEP290K_DPRNA100M':
        source_path = "/root/unirna_L8_E512_STEP290K_DPRNA100M"
    if opts.model_type == 'unirna_L12_E768_STEP210K_DPRNA100M':
        source_path = "/root/unirna_L12_E768_STEP210K_DPRNA100M"
    if opts.model_type == 'unirna_L16_E1024_DPRNA500M_STEP400K':
        source_path = "/root/unirna_L16_E1024_DPRNA500M_STEP400K"
    if opts.model_type == 'unirna_L24_E1280_STEP180K_DPRNA500M':
        source_path = "/root/unirna_L24_E1280_STEP180K_DPRNA500M"

    tokenizer = AutoTokenizer.from_pretrained(source_path)
    model = AutoModel.from_pretrained(source_path)

    sequences = []
    with open(opts.input_data.get_full_path(), "r") as f:
        current_seq = ''
        for line in f:
            if line.startswith('>'):
                if current_seq != '':
                    sequences.append(current_seq)
                current_seq = ''
            else:
                current_seq += line.strip()
        sequences.append(current_seq)


    if opts.output_attentions is True:
        attentions = True
    else:
        attentions = False

    outputs = {}
    for seq in sequences:
        token = tokenizer(seq, return_tensors="pt")
        output = model(**token, output_attentions=attentions)
        outputs[seq] = output

    with open("./outputs/result.pickle", "wb") as f:
        pickle.dump(outputs, f)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(to_runner(
        SingleOptions,
        extract_embeding_runner,
        version="0.0.1",
        exception_handler=default_minimal_exception_handler,
    )(sys.argv[1:]))