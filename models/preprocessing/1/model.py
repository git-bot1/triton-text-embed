import base64
import io
import json
import os
from typing import List

import numpy as np
import requests
import triton_python_backend_utils as pb_utils
from transformers import AutoProcessor, AutoTokenizer, T5Tokenizer


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.
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
        # Parse model configs
        model_config = json.loads(args['model_config'])

        print("---> Model config: ", model_config)
        tokenizer_dir = model_config['parameters']['tokenizer_dir'][
            'string_value']

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir,trust_remote_code=True)

        # if isinstance(self.tokenizer, T5Tokenizer):
        #     self.tokenizer_bos_id = self.tokenizer.sp_model.bos_id()

        # if not self.tokenizer.pad_token:
        #     self.tokenizer.pad_token = self.tokenizer.eos_token

        # self.tokenizer_end_id = self.tokenizer.encode(
        #     self.tokenizer.eos_token, add_special_tokens=False)[0]
        # self.tokenizer_pad_id = self.tokenizer.encode(
        #     self.tokenizer.pad_token, add_special_tokens=False)[0]
        # self.vocab_size = self.tokenizer.vocab_size

        

        # Parse model output configs and convert Triton types to numpy types
        output_names = ["INPUT_ID", "TOKEN_TYPE_IDS", "ATTENTION_MASK"]

        for output_name in output_names:
            setattr(
                self,
                output_name.lower() + "_dtype",
                pb_utils.triton_string_to_numpy(
                    pb_utils.get_output_config_by_name(
                        model_config, output_name)['data_type']))

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse.
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
        for idx, request in enumerate(requests):
            # Get input tensors
            query = pb_utils.get_input_tensor_by_name(request,
                                                      'QUERY').as_numpy()
            
            # Take the end_id from the input tensors
            # If not specified, use tokenizer to get end_id
            # end_id = pb_utils.get_input_tensor_by_name(request, 'END_ID')
            # if end_id is not None:
            #     end_id = end_id.as_numpy()
            # else:
            #     end_id = [[self.tokenizer_end_id]] * batch_size

            # # Take the pad_id from the input tensors
            # # If not specified, use tokenizer to get pad_id
            # pad_id = pb_utils.get_input_tensor_by_name(request, 'PAD_ID')
            # if pad_id is not None:
            #     pad_id = pad_id.as_numpy()
            # else:
            #     pad_id = [[self.tokenizer_pad_id]] * batch_size

            # Preprocessing input data.
            # For the LLaVA_OneVision model, num_visual_features is not a fixed value
            input_ids, token_type_ids, attention_mask = self._create_request(query)
            

            # Create output tensors. You need pb_utils.Tensor
            # objects to create pb_utils.InferenceResponse.
            input_id_tensor = pb_utils.Tensor(
                'INPUT_ID', input_ids.astype(self.input_id_dtype))
            token_type_ids_tensor = pb_utils.Tensor(
                'TOKEN_TYPE_IDS', token_type_ids.astype(self.token_type_ids_dtype))
            attention_mask_tensor = pb_utils.Tensor(
                'ATTENTION_MASK', attention_mask.astype(self.attention_mask_dtype))

            inference_response = pb_utils.InferenceResponse(
                    output_tensors=[
                        input_id_tensor, token_type_ids_tensor,
                        attention_mask_tensor
                    ])
            responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')

    def _create_request(self, query):
        # Decode byte strings to regular strings
        texts = [q[0].decode() for q in query]

        # Tokenize with padding and truncation
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="np",
        )

        # Ensure token_type_ids are present even if model doesn't use them
        if "token_type_ids" not in encodings:
            encodings["token_type_ids"] = np.zeros_like(encodings["input_ids"])

        return encodings["input_ids"], encodings["token_type_ids"], encodings["attention_mask"]
