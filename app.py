import argparse
import os

import gradio as gr
import torch
from diffusers.utils import export_to_video, load_image

from skyreelsinfer import TaskType
from skyreelsinfer.offload import OffloadConfig

# Import our SkyReels modules
from skyreelsinfer.skyreels_video_infer import SkyReelsVideoInfer


def generate_video(
    model_selection,
    prompt,
    negative_prompt,
    guidance_scale,
    embedded_guidance_scale,
    height,
    width,
    num_frames,
    num_inference_steps,
    seed,
    gpu_num,
    quant,
    offload,
    high_cpu_memory,
    parameters_level,
    compiler_transformer,
    sequence_batch,
    fps,
    image_input,
):
    """
    Instantiate the SkyReels video inference pipeline using the provided parameters,
    run inference, and export the output as a video file.
    """
    # Validate that height and width are divisible by 16.
    if height % 16 != 0 or width % 16 != 0:
        return "Error: Both height and width must be divisible by 16. Please adjust your resolution."

    # Set task type and model_id based on the selection.
    if model_selection == "Image-to-Video":
        task = TaskType.I2V
        model_id = "Skywork/SkyReels-V1-Hunyuan-I2V"
    else:
        task = TaskType.T2V
        model_id = "Skywork/SkyReels-V1-Hunyuan-T2V"

    # Build kwargs for the inference call.
    kwargs = {
        "prompt": prompt,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "num_inference_steps": num_inference_steps,
        "seed": seed,
        "guidance_scale": guidance_scale,
        "embedded_guidance_scale": embedded_guidance_scale,
        "negative_prompt": negative_prompt,
        "cfg_for": sequence_batch,
    }

    # For image-to-video, require an image input.
    if task == TaskType.I2V:
        if image_input is None:
            return "Error: Please upload an image for image-to-video inference."
        kwargs["image"] = image_input

    # Create the offload configuration.
    offload_config = OffloadConfig(
        high_cpu_memory=high_cpu_memory,
        parameters_level=parameters_level,
        compiler_transformer=compiler_transformer,
    )

    # Instantiate the predictor.
    predictor = SkyReelsVideoInfer(
        task_type=task,
        model_id=model_id,
        quant_model=quant,
        world_size=gpu_num,
        is_offload=offload,
        offload_config=offload_config,
        enable_cfg_parallel=(guidance_scale > 1.0),
    )

    # Run the inference (this might take a while).
    video_frames = predictor.inference(kwargs)

    # Ensure the results directory exists.
    os.makedirs("results", exist_ok=True)
    safe_prompt = "".join(c if c.isalnum() or c in " _-" else "_" for c in prompt)[:50]
    video_path = os.path.join("results", f"{safe_prompt}_{seed}.mp4")

    # Export the generated frames to a video file.
    export_to_video(video_frames, video_path, fps=fps)
    return video_path


def update_image_visibility(selection):
    """
    Update the visibility of the image upload component.
    """
    if selection == "Image-to-Video":
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)


css = """
h4 {
    text-align: center;
    display:block;
}
"""

with gr.Blocks(
    title="SkyReels Video Inference", theme=gr.themes.Ocean(), css=css
) as iface:
    gr.Markdown(
        """
        # SkyReels Video Inference
        Select the model type from the dropdown below. When you select "Image-to-Video", an image upload field will appear.
        **Note:** Height and width must be divisible by 16.
        """
    )

    with gr.Row():
        model_selection = gr.Dropdown(
            choices=["Text-to-Video", "Image-to-Video"],
            value="Text-to-Video",
            label="Model Selection",
            info="Choose the model type.",
        )

    # Update image input visibility based on the model selection.
    # model_selection.change(update_image_visibility, inputs=model_selection, outputs=[])
    # Instead, directly bind the output to image_input below.

    with gr.Row():
        prompt = gr.Textbox(
            value="FPS-24, A 3D model of a 1800s victorian house.",
            label="Prompt",
            placeholder="Enter your prompt here...",
        )
        negative_prompt = gr.Textbox(
            value="Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion",
            label="Negative Prompt",
        )

    with gr.Row():
        guidance_scale = gr.Slider(0.0, 10.0, value=6.0, label="Guidance Scale")
        embedded_guidance_scale = gr.Slider(
            0.0, 10.0, value=1.0, label="Embedded Guidance Scale"
        )

    with gr.Row():
        height = gr.Slider(
            minimum=16, maximum=1080, step=16, value=544, label="Height (px)"
        )
        width = gr.Slider(
            minimum=16, maximum=1920, step=16, value=960, label="Width (px)"
        )
        num_frames = gr.Number(value=97, label="Number of Frames")
        num_inference_steps = gr.Number(value=30, label="Inference Steps")

    with gr.Row():
        seed = gr.Number(value=42, label="Seed")
        gpu_num = gr.Number(value=1, label="GPU Count (world_size)")

    with gr.Row():
        quant = gr.Checkbox(value=True, label="Enable Quantization (FP8 weight-only)")
        offload = gr.Checkbox(value=True, label="Enable Offload")
        high_cpu_memory = gr.Checkbox(value=True, label="High CPU Memory")
        parameters_level = gr.Checkbox(value=False, label="Parameters-level Offload")
        compiler_transformer = gr.Checkbox(value=False, label="Compile Transformer")
        sequence_batch = gr.Checkbox(value=False, label="Sequence Batch Mode")

    with gr.Row():
        fps = gr.Number(value=24, label="FPS")

    # Image input is only shown when "Image-to-Video" is selected.
    image_input = gr.Image(
        type="pil", label="Input Image (for Image-to-Video)", visible=False
    )

    # Bind the model_selection change to update the image_input visibility.
    model_selection.change(
        update_image_visibility, inputs=model_selection, outputs=image_input
    )

    generate_button = gr.Button("Generate Video")
    output_video = gr.Video(label="Generated Video", format="mp4")

    generate_button.click(
        generate_video,
        inputs=[
            model_selection,
            prompt,
            negative_prompt,
            guidance_scale,
            embedded_guidance_scale,
            height,
            width,
            num_frames,
            num_inference_steps,
            seed,
            gpu_num,
            quant,
            offload,
            high_cpu_memory,
            parameters_level,
            compiler_transformer,
            sequence_batch,
            fps,
            image_input,
        ],
        outputs=output_video,
    )
    gr.Markdown(
        "#### Based on [SkyReels-V1-Hunyuan](https://github.com/SkyworkAI/SkyReels-V1)"
    )
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--listen",
        action="store_true",
        help="If set, the Gradio app will listen on all network interfaces (0.0.0.0) to be accessible on the local network.",
    )
    parser.add_argument(
        "extra_args",
        nargs="*",
        help="Any number of additional arguments. Use `gradio-` prefix to set corresponding Gradio args, e.g. `--gradio-server_name=0.0.0.0`.",
    )
    args = parser.parse_args()

    gradio_kwargs = {}
    for arg in args.extra_args:
        if not arg.startswith("--gradio-"):
            continue
        arg = arg.removeprefix("--gradio-")
        try:
            key, value = arg.split("=", 1)
        except ValueError:
            # bool arg given
            key, value = arg, True
        gradio_kwargs[key] = value

    if args.listen:
        gradio_kwargs["server_name"] = "0.0.0.0"

    if "auth" in gradio_kwargs:
        try:
            username, password = gradio_kwargs["auth"].split(":", 1)
        except ValueError as exc:
            raise ValueError(
                "Username and password must be provided in the format --gradio-auth='username:password'"
            ) from exc
        gradio_kwargs["auth"] = (username, password)

    print("Gradio args:", gradio_kwargs)
    iface.launch(**gradio_kwargs)
