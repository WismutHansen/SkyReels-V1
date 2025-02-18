import argparse
import os

import gradio as gr
from diffusers.utils import export_to_video, load_image

import torch

# Import our SkyReels modules
from skyreelsinfer.skyreels_video_infer import SkyReelsVideoInfer
from skyreelsinfer import TaskType
from skyreelsinfer.offload import OffloadConfig


def generate_video(
    task_type,
    model_id,
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
    This function instantiates the SkyReels video inference pipeline using the provided parameters,
    runs inference, and exports the output as a video file.
    """
    # Determine the task type based on the dropdown selection.
    task = TaskType.I2V if task_type == "i2v" else TaskType.T2V

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

    # For image-to-video (i2v), the user must upload an image.
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
    # Create a temporary filename based on the prompt and seed.
    safe_prompt = "".join(c if c.isalnum() or c in " _-" else "_" for c in prompt)[:50]
    video_path = os.path.join("results", f"{safe_prompt}_{seed}.mp4")

    # Export the generated frames to a video file.
    export_to_video(video_frames, video_path, fps=fps)
    return video_path


# Build the Gradio interface.
with gr.Blocks(title="SkyReels Video Inference") as iface:
    gr.Markdown(
        """
        # SkyReels Video Inference
        This interface uses the SkyReels model for text-to-video (t2v) or image-to-video (i2v) generation.
        """
    )

    with gr.Row():
        task_type = gr.Dropdown(
            choices=["t2v", "i2v"],
            value="t2v",
            label="Task Type",
            info="Choose 't2v' for text-to-video or 'i2v' for image-to-video.",
        )
        model_id = gr.Textbox(
            value="Skywork/SkyReels-V1-Hunyuan-T2V",
            label="Model ID",
            info="For i2v, you can switch to 'Skywork/SkyReels-V1-Hunyuan-I2V' if needed.",
        )

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
        height = gr.Number(value=544, label="Height (px)")
        width = gr.Number(value=960, label="Width (px)")
        num_frames = gr.Number(value=97, label="Number of Frames")
        num_inference_steps = gr.Number(value=30, label="Inference Steps")

    with gr.Row():
        seed = gr.Number(value=42, label="Seed")
        gpu_num = gr.Number(value=1, label="GPU Count (world_size)")

    with gr.Row():
        quant = gr.Checkbox(value=False, label="Enable Quantization (FP8 weight-only)")
        offload = gr.Checkbox(value=False, label="Enable Offload")
        high_cpu_memory = gr.Checkbox(value=False, label="High CPU Memory")
        parameters_level = gr.Checkbox(value=False, label="Parameters-level Offload")
        compiler_transformer = gr.Checkbox(value=False, label="Compile Transformer")
        sequence_batch = gr.Checkbox(value=False, label="Sequence Batch Mode")

    with gr.Row():
        fps = gr.Number(value=24, label="FPS")

    # For image-to-video, allow image upload.
    image_input = gr.Image(type="pil", label="Input Image (for i2v only)", visible=True)

    generate_button = gr.Button("Generate Video")

    output_video = gr.Video(label="Generated Video", format="mp4")

    generate_button.click(
        generate_video,
        inputs=[
            task_type,
            model_id,
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--listen",
        action="store_true",
        help="If set, the Gradio app will listen on all network interfaces (0.0.0.0) to be accessible on the local network.",
    )
    args = parser.parse_args()

    iface.launch(server_name="0.0.0.0" if args.listen else None)
