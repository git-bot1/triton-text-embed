HF_MODEL_REPO=~/tensorrt/text_embed/hf_model
# HF_MODEL_PATH=mixedbread-ai/mxbai-embed-large-v1
HF_MODEL_PATH=nomic-ai/nomic-embed-text-v1
HF_USERNAME=""
HF_TOKEN=""

[ -d $HF_MODEL_REPO ] && rm -rf $HF_MODEL_REPO
mkdir -p $HF_MODEL_REPO && cd $HF_MODEL_REPO

sudo apt-get update -y && sudo apt-get install git-lfs -y --no-install-recommends
git lfs install

GIT_ASKPASS=echo git clone https://$HF_USERNAME:$HF_TOKEN@huggingface.co/$HF_MODEL_PATH .
