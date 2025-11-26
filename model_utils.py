import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import numpy as np
import os


def load_model_and_tokenizer(model_name):
    assert model_name is not None
    model_dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype = model_dtype)

    MODEL_CONFIG={"n_heads":model.config.num_attention_heads,
                    "n_layers":model.config.num_hidden_layers,
                    "resid_dim":model.config.hidden_size,
                    "name_or_path":model.config._name_or_path,
                    "attn_hook_names":[f'model.layers.{layer}.self_attn.o_proj' for layer in range(model.config.num_hidden_layers)],
                    "layer_hook_names":[f'model.layers.{layer}' for layer in range(model.config.num_hidden_layers)],
                    "mlp_hook_names":[f'model.layers.{layer}.mlp' for layer in range(model.config.num_hidden_layers)]
                    }
    return model, tokenizer, MODEL_CONFIG

def set_seed(seed):
    # Random seed
    random.seed(seed)

    # Numpy seed
    np.random.seed(seed)

    # Torch seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # os seed
    os.environ['PYTHONHASHSEED'] = str(seed)