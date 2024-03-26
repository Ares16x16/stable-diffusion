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


def calculate_grid_size(numOfImg):
    if numOfImg == 1:
        return 1, 1
    elif numOfImg == 2:
        return 1, 2
    else:
        factors = []
        for i in range(1, int(numOfImg**0.5) + 1):
            if numOfImg % i == 0:
                factors.append((i, numOfImg // i))

        closest = min(factors, key=lambda f: abs(f[0] - f[1]))
        return closest


if __name__ == "__main__":
    numOfImg = 1
    guidance_scale = 7.5  # 7 - 8.5
    num_inference_steps = 800
    height = 512
    width = 512

    prompts = [
        "A digital illustration of a steampunk library with clockwork machines, 4k, detailed, trending in artstation, fantasy vivid colors"
    ] * numOfImg

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)
    # pipe.enable_sequential_cpu_offload()

    # pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    images = pipe(
        prompt=prompts,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        height=height,
        width=width,
    ).images

    # Grid images
    row, col = calculate_grid_size(numOfImg)
    grid = image_grid(images, rows=row, cols=col)
    grid.save(
        f"steampunk library,guidance_scale{guidance_scale},steps{num_inference_steps}.png"
    )

    # Single images
    """for i, image in enumerate(images):
        save_image(image, str(i) + ".png")"""
