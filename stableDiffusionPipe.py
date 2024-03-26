import torch
from diffusers import (
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
    AutoencoderKL,
    DDPMScheduler,
)
from PIL import Image

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def save_image(image, filename):
    image.save(filename)


if __name__ == "__main__":
    numOfImg = 1

    prompts = [""] * numOfImg

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)
    # pipe.enable_sequential_cpu_offload()

    # pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    """generator = [
        torch.Generator("cuda").manual_seed(0) for _ in range(len(prompts))
    ]"""

    images = pipe(
        prompt=prompts,
        guidance_scale=7.5,  # 7 - 8.5
        # generator=generator,
        num_inference_steps=50,
        height=512,
        width=512,
    ).images

    # Grid images
    grid = image_grid(images, rows=1, cols=1)
    grid.save(f"a.png")

    # Single images
    """for i, image in enumerate(images):
        save_image(image, str(i) + ".png")"""
