from dp.launching.cli import to_runner, default_minimal_exception_handler
from dp.launching.typing import (
    BaseModel,
    Optional,
    Field,
    InputFilePath,
    String,
    Enum,
    OutputDirectory,
)
from rna_app.core.extract_embedding import extract_embedding


class ModelTypeOptions(String, Enum):
    unirna_L8_E512_STEP290K_DPRNA100M = "unirna_L8_E512_STEP290K_DPRNA100M"
    unirna_L12_E768_STEP210K_DPRNA100M = "unirna_L12_E768_STEP210K_DPRNA100M"
    unirna_L16_E1024_DPRNA500M_STEP400K = "unirna_L16_E1024_DPRNA500M_STEP400K"
    unirna_L24_E1280_STEP180K_DPRNA500M = "unirna_L24_E1280_STEP180K_DPRNA500M"


class SingleOptions(BaseModel):
    """
    input file set
    """

    input_data: InputFilePath = Field(default=None, description="Upload fasta file")
    pretrained: ModelTypeOptions = Field(
        default="unirna_L16_E1024_DPRNA500M_STEP400K",
        title="Pretrained Weights",
        description="Choose which unirna weights to use. The default is 'unirna_L16_E1024_DPRNA500M_STEP400K'.",
    )
    output_dir: OutputDirectory = Field(
        default="output", description="Output directory"
    )
    output_attentions: Optional[bool] = Field(
        default=True,
        title="Output Attentions",
        description="Decide whether to output attentions or not. If set to True, the model will output attention weights. The default is True.",
    )


def main(opts: SingleOptions) -> int:
    extract_embedding(
        in_data=opts.input_data.get_full_path(),
        output_dir=opts.output_dir.get_full_path(),
        pretrained=opts.pretrained.split("_")[1],
        output_attentions=opts.output_attentions,
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(
        to_runner(
            SingleOptions,
            main,
            version="0.0.1",
            exception_handler=default_minimal_exception_handler,
        )(sys.argv[1:])
    )
