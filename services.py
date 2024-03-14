import schemas as _schemas
import os
from PIL import Image
from io import BytesIO
import numpy as np
import uuid
import random
from pkg_resources import parse_version
import base64
import cv2
import torch
from diffusers import StableDiffusionImg2ImgPipeline
import subprocess
from tempfile import mkstemp
from shutil import move, copymode
from os import fdopen, remove


TEMP_PATH = 'temp'
MODEL_PATH = os.getenv('MODEL_PATH')
if MODEL_PATH is None:
    MODEL_PATH = 'weights/realisticVisionV60B1_v20Novae.safetensors'

# Helper functions
def set_seed():
    seed = random.randint(42,4294967295)
    return seed


def create_temp():
    if not os.path.exists(TEMP_PATH):
        os.makedirs(TEMP_PATH)


def remove_temp_image(id):
    os.remove(TEMP_PATH + '/' + id + '_input.jpg')
    os.remove(TEMP_PATH + '/' + id + '_generated.jpg')
    os.remove(TEMP_PATH + '/' + id + '_out.jpg')


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
    

def create_pipeline(model_path):
    # Create the pipe 
    pipe = StableDiffusionImg2ImgPipeline.from_single_file(
        model_path, 
        revision="fp16", 
        torch_dtype=torch.float16
        )
    
    # pipe.load_lora_weights(pretrained_model_name_or_path_or_dict="weights/lora_disney.safetensors", adapter_name="disney")

    if torch.backends.mps.is_available():
        device = "mps"
    else: 
        device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe.to(device)
    
    return pipe

pipe = create_pipeline(MODEL_PATH)
# Update the paths in submodule
replace("facefusion/facefusion/core.py", "available_frame_processors = list_directory('facefusion/processors/frame/modules')",
        "available_frame_processors = list_directory('facefusion/facefusion/processors/frame/modules')")
replace("facefusion/facefusion/core.py", "available_ui_layouts = list_directory('facefusion/uis/layouts')",
        "available_ui_layouts = list_directory('facefusion/facefusion/uis/layouts')")
replace("facefusion/facefusion/core.py", "available_frame_processors = list_directory('facefusion/processors/frame/modules')",
        "available_frame_processors = list_directory('facefusion/facefusion/processors/frame/modules')")

async def generate_image(pregnancyCreate: _schemas.PregnancyCreate) -> Image:
    temp_id = str(uuid.uuid4())
    create_temp()

    generator = torch.Generator().manual_seed(set_seed()) if float(pregnancyCreate.seed) == -1 else torch.Generator().manual_seed(int(pregnancyCreate.seed))
    init_image = Image.open(BytesIO(base64.b64decode(pregnancyCreate.encoded_base_img[0])))
    aspect_ratio = init_image.width / init_image.height
    target_height = round(pregnancyCreate.img_width / aspect_ratio)

    # Resize the image
    if parse_version(Image.__version__) >= parse_version('9.5.0'):
        resized_image = init_image.resize((pregnancyCreate.img_width, target_height), Image.LANCZOS)
    else:
        resized_image = init_image.resize((pregnancyCreate.img_width, target_height), Image.ANTIALIAS)

    resized_image.save(TEMP_PATH + '/' + temp_id + '_input.png')

    # Final prompt
    prompt = """
        photo of a nine months pregnant woman, smiling, detailed (wrinkles, blemishes, folds, moles, viens, 
        pores, skin imperfections:1.1), highly detailed glossy eyes, (looking at the camera), 
        specular lighting, dslr, ultra quality, sharp focus, tack sharp, dof, film grain, 
        centered, Fujifilm XT3
    """
    negative_prompt = """
        naked, nude, out of frame, tattoo, b&w, sepia, (blurry un-sharp fuzzy un-detailed skin:1.4), 
        (twins:1.4), (geminis:1.4), (wrong eyeballs:1.1), (cloned face:1.1), (perfect skin:1.2), 
        (mutated hands and fingers:1.3), disconnected hands, disconnected limbs, amputation, 
        (semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, doll, overexposed, photoshop, oversaturated:1.4) 
    """

    image: Image = pipe(prompt,
                                image=resized_image, strength=pregnancyCreate.strength,
                                negative_prompt=negative_prompt, 
                                guidance_scale=pregnancyCreate.guidance_scale, 
                                num_inference_steps=pregnancyCreate.num_inference_steps, 
                                generator = generator,
                                cross_attention_kwargs={"scale": pregnancyCreate.strength}
                                ).images[0]

    if not image.getbbox():
        image: Image = pipe(prompt,
                                    image=resized_image, strength=pregnancyCreate.strength + 0.1,
                                    negative_prompt=negative_prompt,
                                    guidance_scale=pregnancyCreate.guidance_scale, 
                                    num_inference_steps=pregnancyCreate.num_inference_steps, 
                                    generator = generator,
                                    cross_attention_kwargs={"scale": pregnancyCreate.strength}
                                    ).images[0]
        
    image.save(TEMP_PATH + '/' + temp_id + '_generated.png')

    # Swap the input face with the generated image
    subprocess.call(['python', 'facefusion/run.py', '-s', '{}'.format(TEMP_PATH + '/' + temp_id + '_input.png'), 
                      '-t', '{}'.format(TEMP_PATH + '/' + temp_id + '_generated.png'),
                      '-o', '{}'.format(TEMP_PATH + '/' + temp_id + '_out.png'),
                      '--headless', '--frame-processors', 'face_swapper', 'face_enhancer'])
    final_image = Image.open(TEMP_PATH + '/' + temp_id + '_out.png')
    buffered = BytesIO()
    final_image.save(buffered, format="JPEG")
    encoded_img = base64.b64encode(buffered.getvalue())
    remove_temp_image(temp_id)
    
    return encoded_img
        