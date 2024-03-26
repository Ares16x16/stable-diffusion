import torch
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from diffusers.utils import load_image
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)
from controlnet_aux import OpenposeDetector
from transformers import pipeline


### Functions ###
def calculate_grid_size(num_of_images):
    if num_of_images == 1:
        return 1, 1
    elif num_of_images == 2:
        return 1, 2
    else:
        factors = []
        for i in range(1, int(num_of_images**0.5) + 1):
            if num_of_images % i == 0:
                factors.append((i, num_of_images // i))
        closest = min(factors, key=lambda f: abs(f[0] - f[1]))
        return closest


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


### Class ###
class ControlNetDiffusionPipe:
    def __init__(
        self,
        controlNet_model,
        controlNet_model_id,
        diffusion_model_id,
        pipeline_name,
        prompts,
        input_image_url,
        output_image_path,
    ):
        self.pipeline_name = pipeline_name
        self.controlNet_model = controlNet_model
        self.controlNet_model_id = controlNet_model_id
        self.diffusion_model_id = diffusion_model_id
        self.prompts = prompts
        self.input_image_url = input_image_url
        self.output_image_path = output_image_path
        self.pipe = None

    def load_pipeline(self):
        controlnet = ControlNetModel.from_pretrained(
            self.controlNet_model_id, torch_dtype=torch.float32
        )
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.diffusion_model_id,
            controlnet=controlnet,
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_xformers_memory_efficient_attention()

    def process_cannyEdges_images(self, prompts, input_image_url, output_image_path):
        input_image = load_image(input_image_url)
        input_image = np.array(input_image)
        low_threshold = 100
        high_threshold = 200
        canny_image = cv2.Canny(input_image, low_threshold, high_threshold)

        ### zero out middle columns of canny edges
        zero_start = canny_image.shape[1] // 4
        zero_end = zero_start + canny_image.shape[1] // 2
        canny_image[:, zero_start:zero_end] = 0
        ###

        canny_image = np.stack([canny_image] * 3, axis=2)
        canny_image = Image.fromarray(canny_image)

        num_of_images = len(prompts)
        row, col = calculate_grid_size(num_of_images)
        generator = [torch.manual_seed(2) for _ in range(num_of_images)]

        output = self.pipe(
            prompts,
            canny_image,
            negative_prompt=[
                "monochrome, lowres, bad anatomy, worst quality, low quality"
            ]
            * num_of_images,
            num_inference_steps=20,
            generator=generator,
        )

        grid = image_grid(output.images, row, col)
        grid.save(output_image_path)

    def run(self):
        if self.pipe is None:
            print("Pipeline not loaded.")
            return

        prompt_suffix = ", best quality, extremely detailed"
        promptsToPass = [t + prompt_suffix for t in self.prompts]

        if self.controlNet_model == "canny":
            self.process_cannyEdges_images(
                promptsToPass, self.input_image_url, self.output_image_path
            )
            print(
                f"{self.pipeline_name} completed. Output saved to {self.output_image_path}."
            )
        else:
            print("Control Net Model not founded.")


### Model ID ###
canny = "lllyasviel/sd-controlnet-canny"
sdv15 = "runwayml/stable-diffusion-v1-5"
potatoHead = "sd-dreambooth-library/mr-potato-head"
depth = "lllyasviel/sd-controlnet-depth"


def canny_Diffusion():
    diffusion_pipeline = ControlNetDiffusionPipe(
        controlNet_model="canny",
        controlNet_model_id=canny,
        diffusion_model_id=sdv15,
        pipeline_name="Canny+StableDiffusion",
        prompts=["library, bookshelf, many books, metallic"],
        input_image_url=r"C:\Users\EDWARD\Desktop\ELEC4542\stableDiffusion\controlNet\3.png",
        output_image_path="3c.png",
    )
    diffusion_pipeline.load_pipeline()
    diffusion_pipeline.run()


def openpose_Diffusion():
    refImg = load_image(
        r"C:\Users\EDWARD\Desktop\ELEC4542\stableDiffusion\controlNet\4.jpg"
    )
    model = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    pose = model(refImg)

    controlnet = ControlNetModel.from_pretrained(
        "fusing/stable-diffusion-v1-5-controlnet-openpose", torch_dtype=torch.float32
    )

    model_id = sdv15
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        model_id,
        controlnet=controlnet,
        torch_dtype=torch.float32,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    # generator = [torch.Generator(device="cpu").manual_seed(2)]
    prompt = "cat, best quality, extremely detailed"
    output = pipe(
        prompt,
        pose,
        negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
        # generator=generator,
        num_inference_steps=20,
    )
    grid = image_grid(output.images, 1, 1)
    grid.save("4c.png")


def depth_Diffusion():
    depth_estimator = pipeline("depth-estimation")

    image = load_image(r"C:\Users\EDWARD\Downloads\Z62_4338_01.JPG")

    image = depth_estimator(image)["depth"]
    image = np.array(image)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)

    controlnet = ControlNetModel.from_pretrained(depth, torch_dtype=torch.float32)

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        sdv15, controlnet=controlnet, safety_checker=None, torch_dtype=torch.float32
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    prompt = "colorful, red line surround building, galaxy sky, night, bright street."

    image = pipe(
        prompt,
        image,
        negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
        num_inference_steps=20,
    ).images[0]

    image.save("5a.png")


if __name__ == "__main__":
    # canny_Diffusion()
    # openpose_Diffusion()
    depth_Diffusion()
