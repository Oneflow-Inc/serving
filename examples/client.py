"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import time
import argparse
import numpy as np
import tritonclient.http as httpclient
from PIL import Image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image',
                        required=True,
                        help="the image to classify")
    FLAGS = parser.parse_args()

    triton_client = httpclient.InferenceServerClient(url='127.0.0.1:8000')

    image = Image.open(FLAGS.image)
    image = image.resize((224, 224), Image.LANCZOS)
    image = np.asarray(image)
    image = image / 255
    image = np.expand_dims(image, axis=0)
    image = np.transpose(image, axes=[0, 3, 1, 2])
    image = image.astype(np.float32)

    inputs = []
    inputs.append(httpclient.InferInput('INPUT_0', image.shape, "FP32"))
    inputs[0].set_data_from_numpy(image, binary_data=True)
    outputs = []
    outputs.append(httpclient.InferRequestedOutput('OUTPUT_0', binary_data=True, class_count=3))
    now = time.time()
    results = triton_client.infer("resnet50", inputs=inputs, outputs=outputs)
    print(f"time cost: {time.time() - now}s")
    output_data0 = results.as_numpy('OUTPUT_0')
    print(output_data0)
