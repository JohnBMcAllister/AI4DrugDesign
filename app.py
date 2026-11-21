import gradio as gr
from dotenv import load_dotenv

from tabs import (
    integrated_pipeline,
)

load_dotenv()

with gr.Blocks(css=".pending { opacity: 1 !important; }") as demo:
    integrated_pipeline.create_tab()

if __name__ == "__main__":
    demo.launch()