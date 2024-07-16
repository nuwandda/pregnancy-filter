import schemas as _schemas
import os
from PIL import Image
from io import BytesIO
import numpy as np
import uuid
import random
from pkg_resources import parse_version
from rembg import remove as remove_bg
import base64
import cv2
import torch
# from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import StableDiffusionPipeline
import subprocess
from tempfile import mkstemp
from shutil import move, copymode
from os import fdopen, remove
from deepface import DeepFace
from insightface.app import FaceAnalysis
import torch
from diffusers import DDIMScheduler, StableDiffusionXLPipeline
from ip_adapter.ip_adapter_faceid import IPAdapterFaceID, IPAdapterFaceIDXL


TEMP_PATH = 'temp'
base_model_path = "SG161222/RealVisXL_V4.0"
vae_model_path = "stabilityai/sd-vae-ft-mse"
ip_ckpt = "weights/ip-adapter-faceid_sdxl.bin"

background_prompts = ['park', 'school', 'street', 'amusement']
app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.2)

# Helper functions
def set_seed():
    seed = random.randint(42,4294967295)
    return seed


def create_temp():
    if not os.path.exists(TEMP_PATH):
        os.makedirs(TEMP_PATH)


def remove_temp_image(id):
    os.remove(TEMP_PATH + '/' + id + '_input.png')
    os.remove(TEMP_PATH + '/' + id + '_generated.jpg')
    # os.remove(TEMP_PATH + '/' + id + '_out_transparent.png')
    # os.remove(TEMP_PATH + '/' + id + '_final.png')


def replace(file_path, pattern, subst):
    #Create temp file
    fh, abs_path = mkstemp()
    with fdopen(fh,'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                new_file.write(line.replace(pattern, subst))

    #Copy the file permissions from the old file to the new file
    copymode(file_path, abs_path)

    #Remove original file
    remove(file_path)

    #Move new file
    move(abs_path, file_path)


def add_background_to_transparent_image(transparent_image_path, background_image_path, output_image_path):
    # Open the transparent image
    transparent_image = Image.open(transparent_image_path)

    # Open the background image
    background_image = Image.open(background_image_path)

    # Resize background image to match transparent image dimensions
    background_image = background_image.resize(transparent_image.size, Image.ANTIALIAS)

    # Create a new image with the background color
    new_image = Image.new("RGB", transparent_image.size, (255, 255, 255))

    # Create a mask for the transparent parts of the image
    transparency_mask = transparent_image.convert("L").point(lambda x: 255 if x < 128 else 0)

    # Paste the background image onto the new image using the transparency mask
    new_image.paste(background_image, (0, 0), mask=transparency_mask)

    # Paste the original transparent image over the background image
    new_image.paste(transparent_image, (0, 0), mask=transparent_image)

    # Save the result
    new_image.save(output_image_path)

    print("Background added to the transparent parts of the image.")
    

def create_pipe(device='cuda'):
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )

    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        add_watermarker=False
    )

    ip_model = IPAdapterFaceIDXL(pipe, ip_ckpt, device)

    return ip_model

ip_model = create_pipe()

async def generate_image(pregnancyCreate: _schemas.PregnancyCreate) -> Image:
    temp_id = str(uuid.uuid4())
    create_temp()

    init_image = Image.open(BytesIO(base64.b64decode(pregnancyCreate.encoded_base_img[0])))
    faces = app.get(np.asarray(init_image))
    faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
    # aspect_ratio = init_image.width / init_image.height
    # target_height = round(pregnancyCreate.img_width / aspect_ratio)

    # # Resize the image
    # if parse_version(Image.__version__) >= parse_version('9.5.0'):
    #     resized_image = init_image.resize((pregnancyCreate.img_height, target_height), Image.LANCZOS)
    # else:
    #     resized_image = init_image.resize((pregnancyCreate.img_width, target_height), Image.ANTIALIAS)

    init_image.save(TEMP_PATH + '/' + temp_id + '_input.png')
    objs = DeepFace.analyze(img_path = TEMP_PATH + '/' + temp_id + '_input.png', actions = ['race'])
    theme = random.choice(background_prompts)

    # Final prompt
    prompt = "a full body portrait, a pregnant {} woman in a dress, natural skin, dark shot, in the {}".format(objs[0]['dominant_race'], theme)
    negative_prompt = """
        (octane render, render, drawing, anime, bad photo, bad photography:1.3), 
        (worst quality, low quality, blurry:1.2), (bad teeth, deformed teeth, deformed lips), 
        (bad anatomy, bad proportions:1.1), (deformed iris, deformed pupils), (deformed eyes, bad eyes), 
        (deformed face, ugly face, bad face), (deformed hands, bad hands, fused fingers), 
        morbid, mutilated, mutation, disfigured
    """

    print('Final Prompt: ', prompt)
    image = ip_model.generate(prompt=prompt, negative_prompt=negative_prompt, faceid_embeds=faceid_embeds, 
                               guidance_scale=7.5, num_samples=1, 
                               width=1024, height=1024, num_inference_steps=30)[0]
        
    image.save(TEMP_PATH + '/' + temp_id + '_generated.jpg')

    final_image = Image.open(TEMP_PATH + '/' + temp_id + '_generated.jpg')
    buffered = BytesIO()
    final_image.save(buffered, format="JPEG")
    encoded_img = base64.b64encode(buffered.getvalue())
    remove_temp_image(temp_id)
    
    return encoded_img
        