import argparse
import os
import json
import torch
from tqdm import tqdm
from model_utils import *
from data_utils import *
from prompt_utils import *
from baukit import TraceDict, get_module
import torch.nn as nn
from collections import defaultdict
from transformers import get_cosine_schedule_with_warmup
torch.set_grad_enabled(False)
from peft import LoraConfig, get_peft_model, PeftModel

MODEL_CARD={
    "llama3.1-8b": "meta-llama/Llama-3.1-8B",
    "qwen2.5-7b" : "Qwen/Qwen2.5-7B",
    "gemma3-12b" : "google/gemma-3-12b-pt",
    "llama3.2-1b" : "meta-llama/Llama-3.2-1B",
    "qwen2.5-1.5b" : "Qwen/Qwen2.5-1.5B",
    "gemma3-1b" : "google/gemma-3-1b-pt"
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='llama3.1-8b')
    parser.add_argument("--num_shots_by_class", type=int, default=2)
    parser.add_argument("--result_folder", type=str, default='./final_result')
    parser.add_argument("--data_dir", type=str, default='./dataset')
    parser.add_argument("--dataset_name", type=str, default='banking77')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--demon_selection", type=str, default = "stratify")
    parser.add_argument("--max_generate_length", type=int, default=10)


    args = parser.parse_args()

    set_seed(args.seed)
    model_card = MODEL_CARD[args.model_name]

    res_dir = os.path.join(args.result_folder, args.dataset_name, args.model_name, str(args.seed))

    os.makedirs(res_dir, exist_ok=True)

    train_dataset, valid_dataset, test_dataset, label_list = load_data(args.dataset_name, args.data_dir)
    
    device = 'cuda'
    model, tokenizer, model_config = load_model_and_tokenizer(model_card)
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False

    

    activation_storage = torch.load(os.path.join(res_dir, "head_act.pt"))
    h_status = []
    def intervention_fn_inference(head_act, model_config, act_converter, device, step=0):
        def mix_head_act_inference(output, layer_name, inputs):
            current_layer = int(layer_name.split(".")[2])
            if step==0:
                intervention_embedding = head_act[current_layer,:,:].unsqueeze(0).to(device) #shape: (1,n_heads, head_dim)
                intervention_embedding = (torch.zeros(*intervention_embedding.shape, dtype=torch.bfloat16).to(device), intervention_embedding)
            else:
                intervention_embedding = head_act[current_layer]
            if isinstance(inputs, tuple):
                inputs = inputs[0]
            original_shape = inputs.shape
            new_shape = inputs.size()[:-1] + (model_config['n_heads'], model_config['resid_dim']//model_config['n_heads'])
            inputs = inputs.view(*new_shape) # inputs shape: (batch_size , tokens (n), n_heads, head_dim)
            modified_inputs = inputs.clone()
            single_input = inputs[0, -1:, :, :].permute(1,0,2) #shape: (n_heads, 1, head_dim) ## process one example per one inference run, so batch_size is 1 always
            out, status = act_converter(single_input, intervention_embedding) # out shape : (n_heads, 1, head_dim)
            global h_status
            h_status.append(status)
            modified = out.permute(1,0,2) # (1, n_heads, head_dim)
            modified_inputs[0, -1:] = modified.to(dtype=modified_inputs.dtype) # (batch_size, n_heads, head_dim)
            modified_inputs = modified_inputs.contiguous()
            modified_inputs = modified_inputs.view(*original_shape)
            proj_module = get_module(model, layer_name)
            out_proj = proj_module.weight
            new_output = torch.matmul(modified_inputs, out_proj.T)
            return new_output
        return mix_head_act_inference
    

    act_converter = nn.LSTM(input_size=model_config['resid_dim']//model_config['n_heads'], hidden_size=model_config['resid_dim']//model_config['n_heads'], batch_first=True,  dtype = torch.bfloat16)
    act_converter.to(device)
    best_converter_state = torch.load(os.path.join(res_dir, "best_converter_state.pt"))
    act_converter.load_state_dict(best_converter_state)
    act_converter.eval()


    softmax = nn.Softmax(dim=-1)
    correct_cnt = 0
    act_convert_res = []
    new_line_id = tokenizer.encode("\n\n", add_special_tokens=False)[0]
    with torch.no_grad():
        for test_item in tqdm(test_dataset):
            test_prompt, test_target = create_prompt(demon_pool=None, query = test_item, num_shots_by_class=0, option=None, label_list=label_list, shuffle_label=False, prefixes = {"input":"Question:", "output":"Intent:"})
            test_tokenized_input = tokenizer(test_prompt, return_tensors='pt').to(device)
            kv_cache = None
            pred_seq = []
            act_input = activation_storage
            for t in range(args.max_generate_length):
                intervention_fn = intervention_fn_inference(head_act = act_input, model_config = model_config, act_converter=act_converter, device = device, step=t)
                with TraceDict(model, layers=model_config['attn_hook_names'], edit_output=intervention_fn) as td:              
                    output = model.forward(**test_tokenized_input, use_cache=True, past_key_values = kv_cache)
                    kv_cache = output.past_key_values
                output_logits = output.logits[0,-1]
                pred_token_id = torch.argmax(output_logits, dim=-1)
                pred_seq.append(pred_token_id.item())
                if pred_token_id.item()==new_line_id:
                    break
                test_tokenized_input['input_ids'] = pred_token_id.reshape(1,-1)
                test_tokenized_input['attention_mask'] = None
                act_input = h_status.copy()
                h_status = []
            pred_str = tokenizer.decode(pred_seq).strip()
            pred_str = pred_str.replace(tokenizer.eos_token, "")
            pred_str = pred_str.split("\n\n")[0]
            print(pred_str)
            print(test_target)
            print("____________")
            if pred_str == test_target:
                correct_cnt+=1
            act_convert_res.append({"prompt": test_prompt, "query": test_item["input"], "gt": test_target, "pred": pred_str})
            h_status = []
        interv_acc = correct_cnt/len(test_dataset)
        print(interv_acc)
    #     site = {"acc": interv_acc, "res": act_convert_res}
    # with open(os.path.join(res_dir, "extended_act_converter_result.json"), "w") as f:
    #     json.dump(site, f)