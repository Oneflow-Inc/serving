import time
import numpy as np
import tritonclient.http as httpclient
import argparse
from PIL import Image

"""
example: python text_to_image.py --prompt xx
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Simple demo of image generation.")
    parser.add_argument(
        "--prompt", type=str, default="a photo of an astronaut riding a horse on mars"
    )
    cmd_args = parser.parse_args()
    return cmd_args

if __name__ == "__main__":
    args = parse_args()
    prompt = args.prompt
    inputs = []
    prompt_data = np.array([prompt.encode("utf-8")], dtype=np.object_)
    inputs.append(httpclient.InferInput('INPUT_0', (1,), "BYTES"))
    inputs[0].set_data_from_numpy(prompt_data, binary_data=True)

    outputs = []
    outputs.append(httpclient.InferRequestedOutput('OUTPUT_0'))
   
    triton_client = httpclient.InferenceServerClient(url='127.0.0.1:8000')

    now = time.time()
    results = triton_client.infer("sd15", inputs=inputs, outputs=outputs)
    print(f"time cost: {time.time() - now}s")
    img = results.as_numpy('OUTPUT_0')
    Image.fromarray(img).save(f"{prompt}.png")