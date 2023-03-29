import jax
num_devices = jax.device_count()
device_type = jax.devices()[0].device_kind

print(f"Found {num_devices} JAX devices of type {device_type}.")
assert "TPU" in device_type, "Available device is not a TPU, please select TPU from Edit > Notebook settings > Hardware accelerator"

import numpy as np
import jax
import jax.numpy as jnp

from pathlib import Path
from jax import pmap
from flax.jax_utils import replicate
from flax.training.common_utils import shard
from PIL import Image

from huggingface_hub import notebook_login
from diffusers import FlaxStableDiffusionPipeline
dtype = jnp.bfloat16

pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
    # "CompVis/stable-diffusion-v1-4",
    "/home/subin/diffusers/examples/textual_inversion/textual_inversion_space_out",
    # revision="bf16",
    dtype=dtype,
)
print("[i] Pipeline loaded")

def create_key(seed=0):
    return jax.random.PRNGKey(seed)
rng = create_key(0)
rng = jax.random.split(rng, jax.device_count())

prompt = "A <person>"
prompt = [prompt] * jax.device_count()
prompt_ids = pipeline.prepare_inputs(prompt)
print(prompt_ids.shape)

p_params = replicate(params)
prompt_ids = shard(prompt_ids)
print(prompt_ids.shape)

images = pipeline(prompt_ids, p_params, rng, jit=True, num_inference_steps=50, guidance_scale=27.5)[0]

print("[i] Images were generated")
images = images.reshape((images.shape[0],) + images.shape[-3:])
images = pipeline.numpy_to_pil(images)
def image_grid(imgs, rows, cols):
    w,h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, img in enumerate(imgs): grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid
grid = image_grid(images, 2, 4)
# images.save("cat-backpack.png")
grid.save("cat-backpack_grid.png")