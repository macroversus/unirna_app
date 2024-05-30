from dp.launching.cli import to_runner, default_minimal_exception_handler
from dp.launching.typing import (
    BaseModel,
    Field,
    InputFilePath,
    OutputDirectory,
)
from rna_app.core.lncrna_sublocalization import infer_lncrna_sublocalization
from rna_app._version import __version__

class LazyInferenceOptions(BaseModel):
    input_data: InputFilePath = Field(default=None, description="Config")
    output_dir: OutputDirectory = Field(default="output", description="Output directory")

def main(opts: LazyInferenceOptions) -> int:
    infer_lncrna_sublocalization(
        in_filepath=opts.input_data.get_full_path(),
        output_dir=opts.output_dir.get_full_path(),
    )
    return 0


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
