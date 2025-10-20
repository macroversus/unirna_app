from dp.launching.cli import to_runner, default_minimal_exception_handler
from dp.launching.typing import (
    BaseModel,
    Field,
    InputFilePath,
    String,
    Enum,
    OutputDirectory,
)
from rna_app.core.rna_ss import infer_ss
from rna_app._version import __version__


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
    output_dir: OutputDirectory = Field(
        default="output", description="Output directory"
    )


def main(opts: LazyInferenceOptions):
    infer_ss(
        in_data=opts.input_data.get_full_path(),
        output_dir=opts.output_dir.get_full_path(),
        model_type=(
            "unirna" if opts.model_type == ModelTypeOptions.unirna else "archiveii"
        ),
    )


def to_parser():
    return to_runner(
        LazyInferenceOptions,
        main,
        version=__version__,
        exception_handler=default_minimal_exception_handler,
    )


if __name__ == "__main__":
    import sys

    sys.exit(to_parser()(sys.argv[1:]))
