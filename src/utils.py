import subprocess
import logging
from pathlib import Path
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.prompts import PromptTemplate


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
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
    )
    texts = splitter.split_text(md)

    return texts


"""prompt_qa's template

あなたはプロの研究者です。あなたが得意な専門分野に関する文章を正確に理解し、答えることに努めてください。
もし以下の情報が探している情報に関連していない場合は、そのトピックに関する自身の知識を用いて質問に答えてください。

{context}

質問文:{question}

制約条件: {constraint}

回答(日本語)
"""


def prompt_qa(question, constraint):
    """_Generate prompt for QA_

    Args:
        question (str): question.
        constraint (str): constraint.

    Returns:
        str: prompt.
    """
    DEFAULT_SYSTEM_PROMPT = """
    あなたはプロの研究者です。あなたが得意な専門分野に関する文章を正確に理解し、答えることに努めてください。
    もし以下の情報が探している情報に関連していない場合は、そのトピックに関する自身の知識を用いて質問に答えてください。
    """
    input_prompt = PromptTemplate(
        input_variables=[
            "DEFAULT_SYSTEM_PROMPT",
            "context",
            "question",
            "constraint",
        ],
        template="{DEFAULT_SYSTEM_PROMPT}\n\n {context}\n\
        質問文: {question}\n 制約条件: {constraint}\n 回答(日本語)",
    )
    prompt = input_prompt.format(
        DEFAULT_SYSTEM_PROMPT=DEFAULT_SYSTEM_PROMPT,
        question=question,
        constraint=constraint,
    )
    return prompt


# -------------------------------------------------------------------------------

# if you want to use llama_hub.nougat_ocr, you can use this code.

# from llama_hub.nougat_ocr import PDFNougatOCR

# def pdf2doc(uploadable_pdf):
#     reader = PDFNougatOCR()
#     pdf_path = Path("/path/to/pdf")
#     documents = reader.load_data(pdf_path)
#     text = documents[0].text
#     return text


# if you want to use UnstructuredMarkdownLoader, you can use this code.

# from langchain.document_loaders import UnstructuredMarkdownLoader

# def mmd2doc(mmd_path):
#     loader = UnstructuredMarkdownLoader(Path(mmd_path))
#     documents = loader.l
#     return documents[0]
