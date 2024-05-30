from dp.launching.cli import to_runner, default_minimal_exception_handler
from dp.launching.typing import BaseModel, Field, InputFilePath, String, Enum


class ModelTypeOptions(String, Enum):
    """
    二选一
    """

    unirna = "Type_1 - trained on the Uni-RNA 50% threshold secondary structure dataset"
    archiveii = "Type_2 - trained on the MXfold2 (RNAStralign) dataset"


class LazyInferenceOptions(BaseModel):
    input_data: InputFilePath = Field(default=None, description="Upload fasta file")
    model_type: ModelTypeOptions = Field(
        default="Type_1 - trained on the Uni-RNA 50% threshold secondary structure dataset",
        description="Model type",
    )
    # output_dir: OutputDirectory = Field(default="/output", description="Output directory")
    # output_file: String = Field(default="result.pkl", description="Path to the result file")

def main(opts: LazyInferenceOptions):...


def to_parser():
    return to_runner(
        LazyInferenceOptions,
        main,
        version="0.1.0",
        exception_handler=default_minimal_exception_handler,
    )


if __name__ == "__main__":
    import sys

    sys.exit(to_parser()(sys.argv[1:]))
