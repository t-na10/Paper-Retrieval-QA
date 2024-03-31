import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from src.utils import nougatOCR, text_splitter
import shutil

load_dotenv()


def RQA(pdf, question=""):
    """_Execute RetrievalQA_

    Args:
        pdf_path (str): pdf file path.
        question (str): question.

    Returns:
        str: result.
        str: mmd_path.
    """

    pdf_name = pdf.split("/")[-1]
    pdf_path = f"./data/pdf/{pdf_name}"
    mmd_name = pdf_name.replace(".pdf", ".mmd")
    mmd_path = f"./data/output/{mmd_name}"

    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_type=os.environ["OPENAI_API_KEY"],
    )
    model = ChatOpenAI(
        model="gpt-3.5-turbo-16k",
        temperature=0,
        openai_api_key=os.environ["OPENAI_API_KEY"],
        streaming=True,
    )
    db = Chroma(
        embedding_function=embeddings,
    )

    if not os.path.exists(pdf_path):
        shutil.copyfile(pdf, pdf_path)

    # Convert PDF to Markdown
    if not os.path.exists(mmd_path):
        nougatOCR(pdf_path)

    # Chunking
    texts, title = text_splitter(mmd_path)
    # Vector Store
    db.add_documents(texts)

    # Prompt
    DEFAULT_PROMPT = f"あなたはプロの研究者です。あなたが得意な専門分野に関する文章を正確に理解し、\
        質問文に答えることに努めてください。与えられる論文のタイトルは {title} です。"

    RETRIEVAL_PROMPT = """
    与えられる論文の内容は、以下の通りです。
    {context}

    質問文: {question}
    回答(日本語)
    """
    PROMPT = DEFAULT_PROMPT + RETRIEVAL_PROMPT
    prompt_qa = PromptTemplate(
        template=PROMPT,
        input_variables=["context", "question"],
    )

    chain_type_kwargs = {"prompt": prompt_qa}

    qa = RetrievalQA.from_chain_type(
        retriever=db.as_retriever(),
        llm=model,
        chain_type="stuff",
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=False,
    )
    answer = qa(question)
    return answer["result"]
