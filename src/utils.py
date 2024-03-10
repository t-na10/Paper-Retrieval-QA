import subprocess
import logging
from pathlib import Path
from langchain.text_splitter \
    import MarkdownHeaderTextSplitter, SpacyTextSplitter


# License:
# Nougat codebase is licensed under MIT.
# Nougat model weights are licensed under CC-BY-NC.
# reference: https://github.com/facebookresearch/nougat

# model tag : 0.1.0-small, 0.1.0-base

def nougatOCR(pdf_path, output_dir="./data/output/", model="0.1.0-small"):
    """_Convert a PDF file to an MMD(Markdown) file using nougatOCR_

    Args:
        pdf_path (str): pdf file path.
        output_dir (str, optional): Defaults to "./data/output/".
        model (str, optional): Defaults to "0.1.0-small".

    Raises:
        RuntimeError: _description_

    Returns:
        str: subprocess result(standard output).
    """
    cli_command = [
        "nougat",
        str(Path(pdf_path)),
        "-o",
        str(Path(output_dir)),
        "-m",
        model,
    ]
    try:
        result = subprocess.run(
            cli_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        result.check_returncode()
        return result.stdout

    except subprocess.CalledProcessError as e:
        logging.error(
            f"Nougat OCR command failed with return code{e.returncode}:\
                {e.stderr}"
        )
        raise RuntimeError("Nougat OCR command failed.") from e


# License:
# Spacy is licensed under MIT.
# 'en_core_web_sm' is licensed under MIT.
# reference: https://spacy.io/usage/models

# The process this time assumes English.
# The pipeline of SpacyTextSplitter uses 'en_core_web_sm'
# For Japanese, use 'ja_core_news_sm'
# If an error occurs, run 'python -m spacy download en_core_web_sm'

def text_splitter(path):
    """_Split a Markdown file by headers_

    Args:
        path (str): markdown file path.

    Returns:
        list: list of splitted texts.
    """
    with open(path) as f:
        md = f.read()
    headers_to_split_on = [
        ("#", "title"),
        ("##", "chapter"),
        ("###", "section"),
        ("####", "subsection"),
        ("######", "abstract"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
    )
    texts = markdown_splitter.split_text(md)
    title = texts[0].metadata['title']
    try:
        text_splitter = SpacyTextSplitter(
            separator='.',
            pipeline='en_core_web_sm',
            max_length=50000,
        )

        texts = text_splitter.split_documents(texts)
    except NameError:
        cli_command = ["pip", "install", "--upgrade", "--quiet", "spacy"]
        subprocess.run(cli_command)
        cli_command = ["python", "-m", "spacy", "download", "en_core_web_sm"]
        subprocess.run(cli_command)
        text_splitter = SpacyTextSplitter(
            separator='.',
            pipeline='en_core_web_sm',
            max_length=10000,
        )
        texts = text_splitter.split_documents(texts)
    except OSError:
        cli_command = ["python", "-m", "spacy", "download", "en_core_web_sm"]
        subprocess.run(cli_command)
        text_splitter = SpacyTextSplitter(
            separator='.',
            pipeline='en_core_web_sm',
            max_length=10000,
        )
        texts = text_splitter.split_documents(texts)
    return texts, title
