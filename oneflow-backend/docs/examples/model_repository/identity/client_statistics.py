import numpy as np
import tritonclient.http as httpclient


if __name__ == '__main__':
    triton_client = httpclient.InferenceServerClient(url='127.0.0.1:8000')
    statistics = triton_client.get_inference_statistics()
    print(statistics)
