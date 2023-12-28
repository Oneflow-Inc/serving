# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils
import oneflow as flow
from os import path
import numpy as np
import json


class TritonPythonModel:

    def initialize(self, args):
        """
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # parse model_config.
        self.model_config = json.loads(args["model_config"])

        # get input/output number
        self.in_num = len(self.model_config["input"])
        self.out_num = len(self.model_config["output"])

        # get output dtype
        self.output_dtype_ls = []
        for i in range(self.out_num):
            # get output configuration
            output_config = pb_utils.get_output_config_by_name(self.model_config, f"OUTPUT_{i}")

            # convert trition types to numpy types
            output_dtype = pb_utils.triton_string_to_numpy(
                output_config["data_type"]
            )

            # add to ls
            self.output_dtype_ls.append(output_dtype)

        # load oneflow model
        self.model_name = args["model_name"]
        self.model = flow.nn.Graph()
        self.model.load_runtime_state_dict((flow.load(path.join(args["model_repository"], "1", "model"))))
        # import pdb;pdb.set_trace()
        # args['model_instance_device_id'] = "1,2,3" 纯纯字符串
        # device_ls = [int(x.strip()) for x in args['model_instance_device_id'].split(',')]
        # self.model.to(flow.device(f"cuda:device_ls[0]}"))
    

    def execute(self, requests):
        """
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            # get triton input
            triton_in_ls = [pb_utils.get_input_tensor_by_name(request, f"INPUT_{i}") \
                             for i in range(self.in_num) \
            ]

            # infer
            # get oneflow input
            numpy_in_ls = [x.as_numpy() for x in triton_in_ls]
            oneflow_in_ls = [flow.tensor(x, device=flow.device("cuda:1")) for x in numpy_in_ls]
            
            oneflow_out_ls = self.model(*oneflow_in_ls)
            if self.out_num == 1:
                oneflow_out_ls = [oneflow_out_ls]
            
            # trans oneflow output to triton output
            triton_out_ls = []
            for i in range(self.out_num):
                if isinstance(oneflow_out_ls[i], np.ndarray):
                    out = oneflow_out_ls[i].astype(self.output_dtype_ls[i])
                else:
                    out = oneflow_out_ls[i].numpy().astype(self.output_dtype_ls[i])
                triton_out_ls.append(pb_utils.Tensor(f"OUTPUT_{i}", out))

            # set response
            inference_response = pb_utils.InferenceResponse(
                output_tensors=triton_out_ls
            )
            responses.append(inference_response)

        return responses

    def finalize(self):
        pass