# demonstrate the retrieval QA model with chainlit
# if you want to run this code, you command `chainlit run demo_chat.py`

import chainlit as cl
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

import os
from dotenv import load_dotenv
from src.utils import nougatOCR, text_splitter
import shutil

load_dotenv()

embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_type=os.environ["OPENAI_API_KEY"],
)

model = ChatOpenAI(
    model="gpt-3.5-turbo-16k",
    openai_api_key=os.environ["OPENAI_API_KEY"],
    streaming=True,
)

db = Chroma(
    embedding_function=embeddings,
)


@cl.on_chat_start
async def on_chat_start():
    files = None

    while files is None:
        files = await cl.AskFileMessage(
            max_size_mb=20,
            content="Please upload a PDF file to begin!",
            accept=["application/pdf"],
            raise_on_timeout=False,
        ).send()

    file = files[0]
    pdf_path = f"./data/pdf/{file.name}"
    mmd_path = f"./data/output/{file.name.replace('.pdf', '.mmd')}"
    if not os.path.exists(pdf_path):
        shutil.copyfile(file.path, pdf_path)

    if not os.path.exists(mmd_path):
        # Convert PDF to Markdown
        nougatOCR(pdf_path)

    # Chunking
    texts, title = text_splitter(mmd_path)

    # Vector Store
    db.add_documents(texts)

    cl.user_session.set("db", db)
    cl.user_session.set("title", title)

    await cl.Message(content=f"`{file.name}`の読み込みが完了しました。質問を入力してください。").send()


@cl.on_message
async def main(input_message):

    print("入力されたメッセージ: ", str(input_message))
    db = cl.user_session.get("db")
    title = cl.user_session.get("title")

    # Prompt
    DEFAULT_PROMPT = f'あなたはプロの研究者です。あなたが得意な専門分野に関する文章を正確に理解し、\
        質問文に答えることに努めてください。与えられる論文のタイトルは {title} です。'

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
        callbacks=[
            cl.LangchainCallbackHandler()
        ],
    )
    input_content = str(input_message.content)
    answer = qa(input_content)

    await cl.Message(content=answer["result"]).send()
