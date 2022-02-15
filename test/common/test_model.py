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

import argparse
import numpy as np
import tritonclient.http as httpclient


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="the model to request")
    parser.add_argument(
        "-t", "--target-output", required=True, help="target output(.npy)"
    )
    args = parser.parse_args()

    triton_client = httpclient.InferenceServerClient(url="127.0.0.1:8000")

    image = np.ones((1, 3, 224, 224)).astype(np.float32)
    inputs = []
    inputs.append(httpclient.InferInput("INPUT_0", image.shape, "FP32"))
    inputs[0].set_data_from_numpy(image, binary_data=True)
    outputs = []
    outputs.append(httpclient.InferRequestedOutput("OUTPUT_0", binary_data=True))
    results = triton_client.infer(args.model, inputs=inputs, outputs=outputs)
    output_data0 = results.as_numpy("OUTPUT_0")
    flow_output = np.load(args.target_output)
    assert np.allclose(flow_output, output_data0, rtol=1e-03, atol=1e-03)
