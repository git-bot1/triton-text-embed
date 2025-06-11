TRITON_SERVER_VERSION=25.03

IMAGE_NAME=test-image
CONTAINER_NAME=test-embed

docker stop ${CONTAINER_NAME}
docker rm ${CONTAINER_NAME}
# docker run --name ${CONTAINER_NAME} -d --net host --shm-size=2g \
#     --ulimit memlock=-1 --ulimit stack=67108864 --gpus all \
#     -v ~/tensorrt/text_embed/triton_models:/tensorrt/text_embed/triton_models \
#     ${IMAGE_NAME}

# docker run --gpus all --name ${CONTAINER_NAME} --rm -it --net host \
#     -v ~/tensorrt/text_embed/triton_models:/models \
#     -v ~/tensorrt/text_embed/hf_model:/hf_model \
#     -v ~/tensorrt/text_embed/scripts:/scripts \
#     nvcr.io/nvidia/tritonserver:${TRITON_SERVER_VERSION}-py3 \
#     bash /scripts/run_tritonserver.sh

docker run --gpus all --name ${CONTAINER_NAME} --rm -it --net host \
    -v ~/tensorrt/text_embed/triton_models:/models \
    -v ~/tensorrt/text_embed/hf_model:/hf_model \
    -v ~/tensorrt/text_embed/scripts:/scripts \
    tritonserver3:${TRITON_SERVER_VERSION}-py3-ps \
    bash /scripts/run_tritonserver.sh


# curl -X POST \
#   http://localhost:8000/v2/models/model/infer \
#   -H "Content-Type: application/json" \
#   -d '{
#     "inputs": [
#       {
#         "name": "input_ids",
#         "shape": [1, 16],
#         "datatype": "INT64",
#         "data": [[101, 3945, 1035, 23032, 1024, 2040, 2003, 10294, 2015, 3158, 4315, 5003, 3686, 2078, 1029, 102]]
#       },
#       {
#         "name": "token_type_ids",
#         "shape": [1, 16],
#         "datatype": "INT64",
#         "data": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
#       },
#       {
#         "name": "attention_mask",
#         "shape": [1, 16],
#         "datatype": "INT64",
#         "data": [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
#       }
#     ]
#   }' -o ~/tensorrt/text_embed/output.json

# {"error":"[request id: <id_unknown>] unexpected shape for input 'token_type_ids' for model 'model'. Expected [-1,-1], got [4]. NOTE: Setting a non-zero max_batch_size in the model config requires a batch dimension to be prepended to each input shape. If you want to specify the full shape including the batch dim in your input dims config, try setting max_batch_size to zero. See the model configuration docs for more info on max_batch_size."}

# curl -X POST \
#   http://localhost:8000/v2/models/preprocessing/infer \
#   -H "Content-Type: application/json" \
#   -d '{
#     "inputs": [
#       {
#         "name": "QUERY",
#         "shape": [1, 1],
#         "datatype": "BYTES",
#         "data": [["Hello World"]]
#       }
#     ]
#   }'

# curl -X POST \
#   http://localhost:8000/v2/models/ensemble/infer \
#   -H "Content-Type: application/json" \
#   -d '{
#     "inputs": [
#       {
#         "name": "text_input",
#         "shape": [ 2, 1 ],
#         "datatype": "BYTES",
#         "data": [["search_query: What is TSNE?"], ["search_query: Who is Laurens van der Maaten?"]]
#       },
#       {
#         "name": "normalize_final_emb",
#         "shape": [ 2, 1 ],
#         "datatype": "BOOL",
#         "data": [[false], [false]]
#       }
#     ]
#   }' -o ~/tensorrt/text_embed/output5.json -s -w "%{time_total}\n"

# curl -X POST \
#   http://localhost:8000/v2/models/ensemble/infer \
#   -H "Content-Type: application/json" \
#   -d '{
#     "inputs": [
#       {
#         "name": "text_input",
#         "shape": [ 5, 1 ],
#         "datatype": "BYTES",
#         "data": [["Represent this sentence for searching relevant passages: A man is eating a piece of bread"], ["A man is eating food."], ["A man is eating pasta."], ["The girl is carrying a baby."], ["A man is riding a horse."]]
#       }
#     ]
#   }' -o ~/tensorrt/text_embed/output.json

# {"error":"[request id: <id_unknown>] unexpected shape for input 'text_input' for model 'ensemble'. Expected [-1,1], got [1]. NOTE: Setting a non-zero max_batch_size in the model config requires a batch dimension to be prepended to each input shape. If you want to specify the full shape including the batch dim in your input dims config, try setting max_batch_size to zero. See the model configuration docs for more info on max_batch_size."}

# curl -X POST localhost:8000/v2/models/preprocessing/infer -d '{"QUERY": "Hello world"}'

# curl -X POST localhost:8000/v2/models/ensemble/infer -d '{"text_input": "Hello world"}'
