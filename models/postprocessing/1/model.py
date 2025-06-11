import json

import torch
import numpy as np
import torch.nn.functional as F
import triton_python_backend_utils as pb_utils


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

        # Get Pooling method
        # can be one of ["cls_token", "mean_tokens", "max_tokens", "mean_sqrt_len_tokens", "weightedmean_tokens", "lasttoken"]
        default_pooling_mode = model_config['parameters']['default_pooling_mode']['string_value']
        self.default_pooling_mode = default_pooling_mode

        self.model_name = model_config['parameters']['model_name']['string_value']

        input_names = ["INPUT_EMBEDDINGS", "ATTENTION_MASKS"]
        for input_name in input_names:
          setattr(
            self,
            input_name.lower() + "_dtype",
            pb_utils.triton_string_to_numpy(
                pb_utils.get_input_config_by_name(
                    model_config, input_name)['data_type']))
        
        output_names = ["OUTPUT"]
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
        for idx, request in enumerate(requests):
          embeddings = pb_utils.get_input_tensor_by_name(request,'INPUT_EMBEDDINGS').as_numpy()

          requested_pooling_mode = pb_utils.get_input_tensor_by_name(request, 'CUSTOM_POOLING_MODE')
          if requested_pooling_mode is not None:
            requested_pooling_mode = requested_pooling_mode.as_numpy()[0].decode()
          
          normalize_final_emb = pb_utils.get_input_tensor_by_name(request, 'NORMALIZE_FINAL_EMB')
          if normalize_final_emb is not None:
            normalize_final_emb = normalize_final_emb.as_numpy()[0]
          else:
            normalize_final_emb = True

          used_pooling_mode = requested_pooling_mode or self.default_pooling_mode

          print(f"Requested Pooling Mode: {requested_pooling_mode}; Used Pooling Mode: {used_pooling_mode}")

          # can be one of ["cls_token", "mean_tokens", "max_tokens", "mean_sqrt_len_tokens", "weightedmean_tokens", "lasttoken"]
          if used_pooling_mode == 'cls_token':
            final_emb = embeddings[:, 0]

          elif used_pooling_mode == 'mean_tokens':
            # This logic is from Nomic-embed-text-v1. Verify if other models follow the same
            attention_mask = pb_utils.get_input_tensor_by_name(request,'ATTENTION_MASKS').as_numpy()
            pt_attention_mask = torch.from_numpy(attention_mask).float()
            pt_embeddings = torch.from_numpy(embeddings).float()
            input_mask_expanded = pt_attention_mask.unsqueeze(-1).expand(pt_embeddings.size()).float()
            final_emb = torch.sum(pt_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            # final_emb = F.normalize(final_emb, p=2, dim=1)

          elif used_pooling_mode == 'max_tokens':
            final_emb=[]
          elif used_pooling_mode == 'mean_sqrt_len_tokens':
            final_emb=[]
          elif used_pooling_mode == 'weightedmean_tokens':
            final_emb=[]
          elif used_pooling_mode == 'lasttoken':
            # This logic is from Nomic-embed-code
            attention_mask = pb_utils.get_input_tensor_by_name(request,'ATTENTION_MASKS').as_numpy()
            # torch impl
            # pt_attention_mask = torch.from_numpy(attention_mask).float()
            # pt_embeddings = torch.from_numpy(embeddings).float()
            # sequence_lengths = pt_attention_mask.sum(-1) - 1
            # final_emb = pt_embeddings[torch.arange(pt_embeddings.shape[0]), sequence_lengths]

            # numpy impl
            sequence_lengths = np.sum(attention_mask, axis=1) - 1  
            final_emb = embeddings[np.arange(embeddings.shape[0]), sequence_lengths]

          else:
            raise RuntimeError(f"'{used_pooling_mode}' is not a valid pooling mode.")

          if normalize_final_emb:
            final_emb = F.normalize(final_emb, p=2, dim=1)
          
          output_tensor = pb_utils.Tensor(
              'OUTPUT',
              np.array(final_emb).astype(self.output_dtype))
          outputs = [output_tensor]
          inference_response = pb_utils.InferenceResponse(output_tensors=outputs)
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
