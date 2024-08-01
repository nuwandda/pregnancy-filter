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
from diffusers import StableDiffusionPipeline, DPMSolverSinglestepScheduler, AutoencoderKL
from ip_adapter.ip_adapter_faceid import IPAdapterFaceID
import tomesd
import time


TEMP_PATH = 'temp'
# base_model_path = "weights/realisticVisionV60B1_v51HyperVAE.safetensors"
base_model_path = "/home/bugrahan/Documents/Personal/Project/stable-diffusion-webui/models/Stable-diffusion/realisticVisionV60B1_v51HyperVAE.safetensors"
vae_model_path = "stabilityai/sd-vae-ft-mse"
ip_ckpt = "weights/ip-adapter-faceid_sd15.bin"

background_prompts = ['park', 'school', 'street', 'amusement']
app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.2)

# Helper functions
def set_seed():
    """
    This function generates a random seed within a specified range and returns it.
    The seed is used for reproducibility in random number generation.

    Parameters:
    None.

    Returns:
    int: A random seed within the range [42, 4294967295].
    """
    seed = random.randint(42,4294967295)
    return seed


def create_temp():
    """
    This function creates a temporary directory if it does not already exist.

    Parameters:
    None.

    Returns:
    None. The function creates a directory at the path specified by the TEMP_PATH constant.
    If the directory already exists, the function does nothing.
    """
    if not os.path.exists(TEMP_PATH):
        os.makedirs(TEMP_PATH)


def remove_temp_image(id):
    """
    This function removes temporary images created during the image generation process.

    Parameters:
    id (str): The unique identifier of the image. This identifier is used to construct the file paths of the temporary images.

    Returns:
    None. The function removes the temporary images from the file system.
    """
    os.remove(TEMP_PATH + '/' + id + '_input.png')
    os.remove(TEMP_PATH + '/' + id + '_generated.jpg')
    # os.remove(TEMP_PATH + '/' + id + '_out_transparent.png')
    # os.remove(TEMP_PATH + '/' + id + '_final.png')


def replace(file_path, pattern, subst):
    """
    This function replaces all occurrences of a specified pattern in a file with a new substring.
    It creates a temporary file, writes the modified content to it, copies the file permissions from the original file,
    removes the original file, and moves the temporary file to replace it.

    Parameters:
    file_path (str): The path of the file to be modified.
    pattern (str): The substring to be replaced.
    subst (str): The new substring to replace the pattern.

    Returns:
    None. The function modifies the file in place.
    """
    # Create temp file
    fh, abs_path = mkstemp()
    with fdopen(fh,'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                new_file.write(line.replace(pattern, subst))

    # Copy the file permissions from the old file to the new file
    copymode(file_path, abs_path)

    # Remove original file
    remove(file_path)

    # Move new file
    move(abs_path, file_path)


def add_background_to_transparent_image(transparent_image_path, background_image_path, output_image_path):
    """
    This function adds a background image to a transparent image. The background image is resized to match the dimensions
    of the transparent image, and then pasted onto the new image using a transparency mask.

    Parameters:
    transparent_image_path (str): The file path of the transparent image.
    background_image_path (str): The file path of the background image.
    output_image_path (str): The file path where the resulting image will be saved.

    Returns:
    None. The function saves the resulting image to the specified output file path.
    """
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
    noise_scheduler = DPMSolverSinglestepScheduler(
        use_karras_sigmas=True
    )
    vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
    # pipe = StableDiffusionPipeline.from_pretrained(
    #     base_model_path,
    #     torch_dtype=torch.float16,
    #     scheduler=noise_scheduler,
    #     vae=vae,
    #     feature_extractor=None,
    #     safety_checker=None
    # )

    pipe = StableDiffusionPipeline.from_single_file(
        pretrained_model_link_or_path=base_model_path,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        vae=vae,
        feature_extractor=None,
        safety_checker=None,
    )

    tomesd.apply_patch(pipe, ratio=0.5)
    ip_model = IPAdapterFaceID(pipe, ip_ckpt, device)

    return ip_model

ip_model = create_pipe()


async def generate_image(pregnancyCreate: _schemas.PregnancyCreate) -> Image:
    """
    This function generates an image of a pregnant woman based on the input image and parameters.

    Parameters:
    pregnancyCreate (_schemas.PregnancyCreate): An object containing information about the pregnant woman,
        such as the encoded base image.

    Returns:
    Image: The generated image of the pregnant woman. The image is encoded as a base64 string.
    """
    temp_id = str(uuid.uuid4())
    create_temp()
    start = time.time()

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
    prompt = "a full body portrait, a pregnant {} woman, wearing a dress, detailed face, highly detailed, ultra-realistic, in the {}".format(objs[0]['dominant_race'], theme)
    negative_prompt = """
        (nsfw, naked, nude, deformed iris, deformed pupils, semi-realistic, cgi, 3d, 
        render, sketch, cartoon, drawing, anime, mutated hands and fingers:1.4), 
        (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, 
        extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, 
        ugly, disgusting, amputation, bad dress
    """

    print('Final Prompt: ', prompt)
    image = ip_model.generate(prompt=prompt, negative_prompt=negative_prompt, faceid_embeds=faceid_embeds, 
                               guidance_scale=1.5, num_samples=1, 
                               width=512, height=768, num_inference_steps=30)[0]
    # If NSFW is present
    if not image.getbbox():
        image = ip_model.generate(prompt=prompt, negative_prompt=negative_prompt, faceid_embeds=faceid_embeds, 
                        guidance_scale=1.5, num_samples=1, 
                        width=512, height=768, num_inference_steps=40)[0]
        
    image.save(TEMP_PATH + '/' + temp_id + '_generated.jpg')

    print('Time: ', time.time() - start)

    final_image = Image.open(TEMP_PATH + '/' + temp_id + '_generated.jpg')
    buffered = BytesIO()
    final_image.save(buffered, format="JPEG")
    encoded_img = base64.b64encode(buffered.getvalue())
    remove_temp_image(temp_id)
    
    return encoded_img
        