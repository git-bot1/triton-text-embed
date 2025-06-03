TOKENIZER_DIR=/hf_model
TRITON_MODEL_FOLDER=/custom_triton_models
MODEL_FOLDER=/models
TRITON_MAX_BATCH_SIZE=256
INSTANCE_COUNT=1
MAX_QUEUE_DELAY_MS=0
MAX_QUEUE_SIZE=0
FILL_TEMPLATE_SCRIPT=/scripts/fill_template.py

mkdir $TRITON_MODEL_FOLDER
cp -r $MODEL_FOLDER/* $TRITON_MODEL_FOLDER/

cp $TOKENIZER_DIR/onnx/model.onnx $TRITON_MODEL_FOLDER/model/1/

# get word embedding dimension and the pooling config from the below python script
output=$(python3 /scripts/get_config.py)
read EMBED_DIM POOLING_MODE <<< "$output"

echo "Embedding dimension: $EMBED_DIM"
echo "Pooling methods: $POOLING_MODE"

python3 ${FILL_TEMPLATE_SCRIPT} -i ${TRITON_MODEL_FOLDER}/ensemble/config.pbtxt triton_max_batch_size:${TRITON_MAX_BATCH_SIZE}
python3 ${FILL_TEMPLATE_SCRIPT} -i ${TRITON_MODEL_FOLDER}/preprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},preprocessing_instance_count:${INSTANCE_COUNT}
python3 ${FILL_TEMPLATE_SCRIPT} -i ${TRITON_MODEL_FOLDER}/model/config.pbtxt triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MS},max_queue_size:${MAX_QUEUE_SIZE},word_embedding_dimension:${EMBED_DIM}
python3 ${FILL_TEMPLATE_SCRIPT} -i ${TRITON_MODEL_FOLDER}/postprocessing/config.pbtxt triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},postprocessing_instance_count:${INSTANCE_COUNT},default_pooling_mode:${POOLING_MODE}

pip3 install requests transformers torch

tritonserver --model-repository=${TRITON_MODEL_FOLDER}
