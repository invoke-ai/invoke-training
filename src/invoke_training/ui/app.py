import gradio as gr


def launch_app():
    with gr.Blocks() as demo:
        gr.Markdown(
            """
        # Hello World!
        This is a demo of the Blocks interface. Blocks is a way to create a Gradio interface
        """
        )

    demo.launch()
