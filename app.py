from typing import List
import gradio as gr
import PIL.Image, PIL.ImageOps
import torch
import numpy as np
import torchvision.transforms as T

from src.models.yolov3 import YOLOv3
from src.train import draw_bounding_boxes, decode_predictions_3scales
from src.dataset import ANCHORS, resize_with_padding


device = torch.device("cpu")
model_weight = "weights/checkpoint-best.pth"
label_colors = {"without_mask": (178, 34, 34), "with_mask": (34, 139, 34), "mask_worn_incorrectly": (184, 134, 11)}

model = YOLOv3()
model.load_state_dict(torch.load(model_weight, map_location=device))
model.eval()


def create_combined_image(img: torch.Tensor, results: List[torch.Tensor], mean: List[float] = [0.485, 0.456, 0.406], std: List[float] = [0.229, 0.224, 0.225]):
    batch_size, _, height, width = img.shape
    combined_height = height
    combined_width = width * batch_size
    combined_image = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)

    for i in range(batch_size):
        image = img[i].cpu().permute(1, 2, 0).numpy()
        image = (image * std + mean).clip(0, 1)
        image = (image * 255).astype(np.uint8)
        pred_image = PIL.Image.fromarray(image.copy())
        draw_bounding_boxes(pred_image, results[i], show_conf=True)
        combined_image[:height, i * width:(i + 1) * width, :] = np.array(pred_image)
    return PIL.Image.fromarray(combined_image)


transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def detect_mask(image, conf_threshold: float) -> PIL.Image:
    img_resized, _, _, _ = resize_with_padding(image)
    img_tensor = transform(img_resized)
    with torch.no_grad():
        out_l, out_m, out_s = model(img_tensor.unsqueeze(0))
    results = decode_predictions_3scales(out_l, out_m, out_s, ANCHORS["large"], ANCHORS["medium"], ANCHORS["small"], conf_threshold=conf_threshold)
    combined_image = create_combined_image(img_tensor.unsqueeze(0), results)
    return combined_image


def generate_legend_html_compact() -> str:
    legend_html = """
    <div style="display: flex; flex-wrap: wrap; gap: 10px; justify-content: center;">
    """
    for idx, (label, color) in enumerate(label_colors.items()):
        legend_html += f"""
        <div style="display: flex; align-items: center; justify-content: center; 
                     padding: 5px 10px; border: 1px solid rgb{color}; 
                     background-color: rgb{color}; border-radius: 5px; 
                     color: white; font-size: 12px; text-align: center;">
            {label}
        </div>
        """
    legend_html += "</div>"
    return legend_html


examples = [
    ["assets/examples/image1.jpg"],
    ["assets/examples/image2.jpg"],
    ["assets/examples/image3.jpg"],
    ["assets/examples/image4.jpg"],
    ["assets/examples/image5.jpg"]
]


with gr.Blocks() as demo:
    gr.Markdown("## Mask Detection with YOLOv3")
    with gr.Row():
        with gr.Column():
            pic = gr.Image(label="Upload Human Image", type="pil", height=300, width=300)
            conf_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.9, step=0.01, label="Confidence Threshold")
            with gr.Row():
                with gr.Column(scale=1):
                    predict_btn = gr.Button("Predict")
                with gr.Column(scale=1):
                    clear_btn = gr.Button("Clear")
        
        with gr.Column():
            output = gr.Image(label="Detection", type="pil", height=300, width=300)
            legend = gr.HTML(label="Legend", value=generate_legend_html_compact())

    predict_btn.click(fn=detect_mask, inputs=[pic, conf_slider], outputs=output, api_name="predict")
    clear_btn.click(lambda: (None, None), outputs=[pic, output])
    gr.Examples(examples=examples, inputs=[pic])

demo.launch()