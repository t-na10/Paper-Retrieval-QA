# demonstrate the retrievalQA  with gradio

import gradio as gr
from gradio_pdf import PDF
from src.retrievalQA import RQA

# default values
DEFAULT_QUESTION = "この論文の学術的貢献を要約しなさい"

with gr.Blocks() as demo:
    gr.Markdown(("# Paper Retrieval QA Demo\n"))
    gr.Markdown(
        "論文(PDF)に関する問い合わせ応答（要約・質問）Webアプリ\n\n"
        "## 論文のアップロード\n"
    )
    with gr.Row():
        pdf = PDF(label="対象のPDFファイル(Optional)")
    gr.Markdown(
        ("## プロンプトの作成\n" "プロンプトの内容はここで自由に書き換えてOKです。")
    )
    with gr.Tabs():
        with gr.TabItem("プロンプト"):
            question = gr.Textbox(
                label="質問文",
                placeholder="質問文",
                lines=3,
                value=DEFAULT_QUESTION,
            )
            button = gr.Button("実行")
    result_box = gr.Text(label="回答")

    button.click(RQA, inputs=[pdf, question], outputs=[result_box])

if __name__ == "__main__":
    demo.queue().launch(share=True)
