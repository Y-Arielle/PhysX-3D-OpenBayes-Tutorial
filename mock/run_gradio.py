# run_gradio.py
import gradio as gr
import os
from example_mock import mock_load_models, mock_run

# 初始化（mock 加载）
mock_load_models()

def gradio_run(image, question):
    # 保存临时输入
    os.makedirs("tmp_inputs", exist_ok=True)
    input_path = os.path.join("tmp_inputs", "input.png")
    image.save(input_path)
    outs = mock_run(input_path, question)
    # Gradio 返回视频和 GLB 占位说明文件（作为可下载文件）
    return outs["video"], outs["glb_note"], outs["image"]

title = "PhysX-3D (Mock Demo) - OpenBayes Tutorial"
description = "此演示为 PhysX-3D 的 mock 教程示例，仅用于测试在 OpenBayes 平台构建教程与 Gradio 一键部署。启用真实模型请参见 README。"

demo = gr.Interface(
    fn=gradio_run,
    inputs=[
        gr.Image(type="pil", label="Input Image"),
        gr.Textbox(value="Cushioned seat surface", label="Question / Prompt")
    ],
    outputs=[
        gr.Video(label="Rendered Mock Video"),
        gr.File(label="Model (placeholder note)"),
        gr.Image(label="Rendered Mock Image")
    ],
    title=title,
    description=description,
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8080)
