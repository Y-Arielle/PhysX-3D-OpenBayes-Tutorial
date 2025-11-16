# example_mock.py
import os
from PIL import Image
import imageio

# 这是 mock 版本，仅用于教程示意与 UI 运行。真实推理需替换为原 example.py 中加载模型的逻辑。
print("=== PhysX-3D MOCK example ===")

def mock_load_models():
    # 模拟模型加载
    print("[MOCK] Loading trellis model... done.")
    print("[MOCK] Loading diffusion model... done.")
    return True

def mock_run(image_path, question="Cushioned seat surface"):
    # 模拟推理并生成输出文件
    print(f"[MOCK] Running inference on {image_path} with question='{question}'")
    # 生成一个小的示例图片/视频作为输出（用图像拼接代替渲染结果）
    out_dir = os.path.join("outputs_vis")
    os.makedirs(out_dir, exist_ok=True)
    # 生成 mock 输出图片
    im = Image.new("RGB", (512, 512), color=(73, 109, 137))
    im.save(os.path.join(out_dir, "mock_render.png"))
    # 生成 mock video (重复同一张图)
    frames = [imageio.imread(os.path.join(out_dir, "mock_render.png")) for _ in range(10)]
    imageio.mimsave(os.path.join(out_dir, "mock_affordance.mp4"), frames, fps=4)
    # 生成 mock glb 占位文件（文本提示）
    with open(os.path.join(out_dir, "mock_model_placeholder.txt"), "w") as f:
        f.write("This is a mock placeholder for texture.glb. Replace with real GLB after model download.")
    return {
        "video": os.path.join(out_dir, "mock_affordance.mp4"),
        "image": os.path.join(out_dir, "mock_render.png"),
        "glb_note": os.path.join(out_dir, "mock_model_placeholder.txt"),
    }

if __name__ == "__main__":
    # 运行示例（本地测试）
    mock_load_models()
    # 这里默认使用 example/table.png 作为输入
    sample_image = os.path.join("example", "table.png")
    if not os.path.exists(sample_image):
        # 生成占位输入图片
        from PIL import ImageDraw
        os.makedirs("example", exist_ok=True)
        img = Image.new("RGB", (512,512), color=(200,200,200))
        d = ImageDraw.Draw(img)
        d.text((20,20), "Sample Input (mock)", fill=(0,0,0))
        img.save(sample_image)
    outputs = mock_run(sample_image)
    print("Mock outputs saved to:", outputs)
