from pathlib import Path

from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document


def load_documents(
    data_dir: str | Path = "data",
    required_exts: list[str] | None = None,
) -> list[Document]:
    if required_exts is None:
        required_exts = [".pdf"]

    reader = SimpleDirectoryReader(
        input_dir=str(data_dir),
        required_exts=required_exts,
        filename_as_id=True,
    )
    return reader.load_data(show_progress=True)
