import json
import sys
from pathlib import Path

def get_config():
    config_path = Path("/hf_model/1_Pooling/config.json")

    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        print("Error: Config file not found at ", config_path)
        sys.exit(1)
    
    embedding_dim = config.get("word_embedding_dimension", "unknown")

    if config.get("pooling_mode_cls_token", False):
        pooling_mode = "cls_token"
    elif config.get("pooling_mode_mean_tokens", False):
        pooling_mode = "mean_tokens"
    elif config.get("pooling_mode_max_tokens", False):
        pooling_mode = "max_tokens"
    elif config.get("pooling_mode_mean_sqrt_len_tokens", False):
        pooling_mode = "mean_sqrt_len_tokens"
    elif config.get("pooling_mode_weightedmean_tokens", False):
        pooling_mode = "weightedmean_tokens"
    elif config.get("pooling_mode_lasttoken", False):
        pooling_mode = "lasttoken"

    print(f"{embedding_dim} {pooling_mode}")


if __name__ == "__main__":
    get_config()
