import os
import openai

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from utils import nougatOCR, text_splitter


load_dotenv()


def RQA(pdf_path, question='', chain_type='refine'):
    """_Execute RetrievalQA_

    Args:
        pdf_path (str): pdf file path.
        question (str): question.

    Returns:
        str: result.
    """

    pdf_name = pdf_path.split("/")[-1]
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
    )
    db = Chroma(
        persist_directory="./db",
        embedding_function=embeddings,
    )

    # Prompt
    DEFAULT_CHAT_PROMPT = """
    あなたはプロの研究者です。あなたが得意な専門分野に関する文章を正確に理解し、質問文に答えることに努めてください。

    {context}

    質問文: {question}
    回答(日本語)
    """
    prompt_qa = PromptTemplate(
        template=DEFAULT_CHAT_PROMPT,
        input_variables=["context", "question"],
    )

    # Convert PDF to Markdown
    if not os.path.exists(mmd_path):
        nougatOCR(pdf_path)

    # Chunking
    texts = text_splitter(mmd_path)
    # Vector Store
    db.add_documents(texts)

    # Retriever
    chain_type_kwargs = {"prompt": prompt_qa}
    qa = RetrievalQA.from_chain_type(
        llm=model,
        retriever=db.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
        chain_type=chain_type
    )
    answer = qa(question)
    return answer['result'], answer['source_documents']
