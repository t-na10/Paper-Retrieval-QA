# MIT License

import gradio as gr
from src.retrievalQA import RQA

# default values
DEFAULT_QUESTION = "この論文の学術的貢献を要約しなさい"
DEFAULT_CONSTRAINT = "入力や出力に対する制約条件(Optional)"


if __name__ == "__main__":
    with gr.Blocks() as demo:
        gr.Markdown(("# Paper Retrieval QA Demo\n"))
        gr.Markdown("論文(PDF)に関する問い合わせ応答（要約・質問）を可能にするWebアプリ\n\n"
                    "## 論文のアップロード\n")
        with gr.Row():
            uploadable_txt = gr.File(label="対象のPDFファイル(Optional)",
                                     file_types=["pdf"])
        gr.Markdown(("## プロンプトの作成\n" "プロンプトの内容はここで自由に書き換えてOKです。"))
        with gr.Tabs():
            with gr.TabItem("プロンプト"):
                gr.Markdown(("質問文・制約条件にはそれぞれお好みの内容を記載してください。"))
                all_token_inputs = [
                    gr.Textbox(
                        label="質問文",
                        placeholder="質問文",
                        lines=3,
                        value=DEFAULT_QUESTION,
                        max_lines=15,
                    ),
                    gr.Textbox(
                        label="制約条件",
                        placeholder="制約条件",
                        lines=3,
                        value=DEFAULT_CONSTRAINT,
                        max_lines=15,
                    ),
                ]
                all_token_button = gr.Button("推論")
        out_box = gr.Text(label="回答")

        all_token_button.click(
            RQA, inputs=[uploadable_txt, *all_token_inputs], outputs=[out_box]
        )

    demo.queue().launch(share=True)
