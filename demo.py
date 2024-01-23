# demonstrate the retrievalQA  with gradio

import gradio as gr
from src.retrievalQA import RQA

# default values
DEFAULT_QUESTION = "この論文の学術的貢献を要約しなさい"

if __name__ == "__main__":
    with gr.Blocks() as demo:
        gr.Markdown(("# Paper Retrieval QA Demo\n"))
        gr.Markdown("論文(PDF)に関する問い合わせ応答（要約・質問）を可能にするWebアプリ\n\n"
                    "## 論文のアップロード\n")
        with gr.Row():
            pdf = gr.File(label="対象のPDFファイル(Optional)", file_types=["pdf"])
        gr.Markdown(("## プロンプトの作成\n" "プロンプトの内容はここで自由に書き換えてOKです。"))
        with gr.Tabs():
            with gr.TabItem("プロンプト"):
                gr.Markdown(("質問文はお好みの内容を記載してください。"))
                question = gr.Textbox(
                    label="質問文",
                    placeholder="質問文",
                    lines=3,
                    value=DEFAULT_QUESTION,
                )

                button = gr.Button("推論")
        result_box = gr.Text(label="回答")
        source_box = gr.Text(label="参照元")

        button.click(
            RQA,
            inputs=[pdf, question],
            outputs=[result_box, source_box]
        )

    demo.queue().launch(share=True)
