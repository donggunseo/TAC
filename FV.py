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
torch.set_grad_enabled(False)

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
    parser.add_argument("--n_trials_by_class", type=int, default=3)
    parser.add_argument("--result_folder", type=str, default='./exp1')
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
    for param in model.parameters():
        param.requires_grad = False
    model.to(device)

    def split_activations_by_head(activations, model_config):
        new_shape = activations.size()[:-1] + (model_config['n_heads'], model_config['resid_dim']//model_config['n_heads']) # split by head: + (n_attn_heads, hidden_size/n_attn_heads)
        activations = activations.view(*new_shape)  # (batch_size, n_tokens, n_heads, head_hidden_dim)
        return activations
    
    ## mean_head_activation
    if os.path.exists(os.path.join(res_dir, "head_act.pt")):
        activation_storage = torch.load(os.path.join(res_dir, "head_act.pt"))
    else:
        train_buckets = defaultdict(list)
        for item in train_dataset:
            if item["output"] in label_list:
                train_buckets[item["output"]].append(item)
        activation_storage = []
        num_act_by_class = args.n_trials_by_class
        for l in tqdm(label_list):
            cur_act_cnt = 0
            while cur_act_cnt<num_act_by_class:
                dummy_query = random.choice(train_buckets[l])
                demon_pool = [item for item in train_dataset if item!=dummy_query]
                dummy_prompt, dummy_query_target, _= create_prompt(demon_pool=demon_pool, query = dummy_query, num_shots_by_class=args.num_shots_by_class, option=args.demon_selection, label_list=label_list, shuffle_label=False)
                dummy_tokenized_input = tokenizer(dummy_prompt, return_tensors='pt').to(device)
                dummy_output = model.generate(**dummy_tokenized_input, max_new_tokens = 10, pad_token_id=tokenizer.eos_token_id, eos_token_id = tokenizer.eos_token_id, tokenizer=tokenizer, do_sample=False, temperature = None, top_p = None, stop_strings = ["\n\n"])
                dummy_pred_str = tokenizer.decode(dummy_output.squeeze()[len(dummy_tokenized_input.input_ids.squeeze()):], skip_speical_tokens=True)
                dummy_pred_str = dummy_pred_str.strip()
                dummy_pred_str = dummy_pred_str.replace(tokenizer.eos_token, "")
                dummy_pred_str = dummy_pred_str.split("\n\n")[0]
                if dummy_pred_str==dummy_query_target:
                    torch.cuda.empty_cache()
                    with torch.no_grad() and TraceDict(model, layers=model_config['attn_hook_names'], retain_input=True, retain_output=False) as td:
                        outputs = model.forward(**dummy_tokenized_input)
                    stack_initial = torch.vstack([split_activations_by_head(td[layer].input, model_config) for layer in model_config['attn_hook_names']]).permute(0,2,1,3)
                    cur_activation = stack_initial[:, :, -1, :].cpu().detach()
                    activation_storage.append(cur_activation)
                    cur_act_cnt+=1
                else:
                    continue
        activation_storage = torch.stack(activation_storage)
        activation_storage = torch.mean(activation_storage, dim=0)
        torch.save(activation_storage, os.path.join(res_dir, "head_act.pt"))
    
    #compute indirect effect
    def intervention_fn_rep_act(head_act, model_config, device, edit_layer, edit_head):
        def rep_act(output, layer_name, inputs):
            current_layer = int(layer_name.split('.')[2])
            if current_layer == edit_layer:    
                if isinstance(inputs, tuple):
                    inputs = inputs[0]
                original_shape = inputs.shape
                new_shape = inputs.size()[:-1] + (model_config['n_heads'], model_config['resid_dim']//model_config['n_heads'])
                inputs = inputs.view(*new_shape) # inputs shape: (batch_size , tokens (n), n_heads, head_dim)
                modified_inputs = inputs.clone()

                modified_inputs[:,-1, edit_head, :]=head_act[edit_layer, edit_head].to(device)
                modified_inputs = modified_inputs.contiguous()
                modified_inputs = modified_inputs.view(*original_shape)
                proj_module = get_module(model, layer_name)
                out_proj = proj_module.weight
                new_output = torch.matmul(modified_inputs, out_proj.T)
                return new_output
            else:
                return output
        return rep_act

    if os.path.exists(os.path.join(res_dir, "indirect_effect.pt")):
        indirect_effect = torch.load(os.path.join(res_dir, "indirect_effect.pt"))
    else:
        softmax = nn.Softmax(dim=-1)
        indirect_effect = torch.zeros(model_config['n_layers'], model_config['n_heads'])
        train_buckets = defaultdict(list)
        for item in train_dataset:
            if item["output"] in label_list:
                train_buckets[item["output"]].append(item)
        total_cnt = len(label_list)*args.n_trials_by_class
        for l in tqdm(label_list):
            cur_trial_cnt=0
            while cur_trial_cnt < args.n_trials_by_class:
                dummy_query = random.choice(train_buckets[l])
                demon_pool = [item for item in train_dataset if item!=dummy_query]
                dummy_prompt, dummy_query_target, demonstrations= create_prompt(demon_pool=demon_pool, query = dummy_query, num_shots_by_class=args.num_shots_by_class, option=args.demon_selection, label_list=label_list, shuffle_label=False)
                dummy_tokenized_input = tokenizer(dummy_prompt, return_tensors='pt').to(device)
                dummy_output = model.generate(**dummy_tokenized_input, max_new_tokens = args.max_generate_length, pad_token_id=tokenizer.eos_token_id, eos_token_id = tokenizer.eos_token_id, tokenizer=tokenizer, do_sample=False, temperature = None, top_p = None, stop_strings = ["\n\n"])
                dummy_pred_str = tokenizer.decode(dummy_output.squeeze()[len(dummy_tokenized_input.input_ids.squeeze()):], skip_speical_tokens=True)
                dummy_pred_str = dummy_pred_str.strip()
                dummy_pred_str = dummy_pred_str.replace(tokenizer.eos_token, "")
                dummy_pred_str = dummy_pred_str.split("\n\n")[0]
                if dummy_pred_str==dummy_query_target:
                    torch.cuda.empty_cache()
                    dummy_prompt, dummy_query_target, _ = create_prompt(demon_pool = demonstrations, query = dummy_query, num_shots_by_class=args.num_shots_by_class, option='all', label_list=label_list, shuffle_label=True)
                    dummy_tokenized_input = tokenizer(dummy_prompt, return_tensors='pt').to(device)
                    target_id = tokenizer.encode(": "+dummy_query_target, add_special_tokens = False)[1]
                    with torch.no_grad():
                        prefix_ids = dummy_tokenized_input.input_ids[:,:-1]
                        last_ids = dummy_tokenized_input.input_ids[:,-1:]
                        prefix_mask = dummy_tokenized_input.attention_mask[:,:-1]
                        last_mask = dummy_tokenized_input.attention_mask
                        out_prefill = model(input_ids=prefix_ids, attention_mask=prefix_mask, use_cache=True)
                        prefix_cache=out_prefill.past_key_values
                        for layer_idx in tqdm(range(model_config['n_layers'])):
                            for head_idx in range(model_config['n_heads']):
                                corrupted_output = model(input_ids=last_ids, attention_mask=last_mask, past_key_values=prefix_cache, use_cache=False)
                                corrupted_output = softmax(corrupted_output.logits[0,-1,:]).cpu().detach()
                                intervention_fn = intervention_fn_rep_act(activation_storage, model_config, device, edit_layer=layer_idx, edit_head=head_idx)
                                with TraceDict(model, layers=model_config['attn_hook_names'], edit_output=intervention_fn):              
                                    replaced_output = model(input_ids=last_ids, attention_mask=last_mask, past_key_values=prefix_cache, use_cache=False)
                                replaced_output = softmax(replaced_output.logits[0,-1,:]).cpu().detach()
                                cie = replaced_output[target_id].item()-corrupted_output[target_id].item()
                                indirect_effect[layer_idx, head_idx]+=cie
                    cur_trial_cnt +=1
                else:
                    continue
        indirect_effect = indirect_effect/total_cnt
        torch.save(indirect_effect, os.path.join(res_dir, "indirect_effect.pt"))

                                
    top_n_heads = 102
    values, flat_indices = torch.topk(indirect_effect.view(-1), top_n_heads)
    rows, cols = torch.unravel_index(flat_indices, indirect_effect.shape)
    topk_indices_2d = list(zip(rows.tolist(), cols.tolist()))
    topk_with_values = list(zip(topk_indices_2d, values.tolist()))
    print(topk_with_values)

    fv = torch.zeros(model_config['resid_dim']).to(device)
    head_dim = model_config['resid_dim']//model_config['n_heads']
    for (L,H) in topk_indices_2d:
        out_proj = model.model.layers[L].self_attn.o_proj
        x = torch.zeros(model_config['resid_dim'], dtype = torch.bfloat16)
        x[H*head_dim:(H+1)*head_dim] = activation_storage[L,H]
        d_out = out_proj(x.to(device))
        fv+=d_out
    
    test_edit_layer = 11
    def add_function_vector(edit_layer, fv_vector, device, idx=-1):
        def add_act(output, layer_name):
            current_layer = int(layer_name.split(".")[2])
            if current_layer == edit_layer:
                fv_vector.to(device)
                if isinstance(output, tuple):
                    output[0][:, idx] += fv_vector
                    return output
                else:
                    output[:,idx] += fv_vector
                    return output
            else:
                return output
        return add_act
    
    softmax = nn.Softmax(dim=-1)
    correct_cnt = 0
    first_token_match=0
    fv_res = []
    new_line_id = tokenizer.encode("\n\n", add_special_tokens=False)[0]
    with torch.no_grad():
        for test_item in tqdm(test_dataset[1500:2000]):
            test_prompt, test_target, _ = create_prompt(demon_pool=None, query = test_item, num_shots_by_class=0, option=None, label_list=label_list, shuffle_label=False)
            test_tokenized_input = tokenizer(test_prompt, return_tensors='pt').to(device)
            kv_cache = None
            pred_seq = []
            prob_seq= []
            target_seq = tokenizer.encode(": "+test_target, add_special_tokens=False)[1:]
            for t in range(args.max_generate_length):
                if t==0:
                    intervention_fn = add_function_vector(edit_layer = test_edit_layer, fv_vector = fv, device=device, idx=-1)
                    with TraceDict(model, layers=model_config['layer_hook_names'], edit_output=intervention_fn):              
                        output = model.forward(**test_tokenized_input, use_cache=True, past_key_values = kv_cache)
                        kv_cache = output.past_key_values
                else:
                    output = model.forward(**test_tokenized_input, use_cache=True, past_key_values = kv_cache)
                output_logits = output.logits[0,-1]
                output_prob = softmax(output_logits)
                topk_probs, topk_indices = torch.topk(output_prob, k=10)
                probs_dict = {tokenizer.decode(int(token_id)): float(prob) for token_id, prob in zip(topk_indices, topk_probs)}
                pred_token_id = torch.argmax(output_logits, dim=-1)
                pred_seq.append(pred_token_id.item())
                prob_seq.append(probs_dict)
                if pred_token_id.item()==new_line_id or pred_token_id.item()==tokenizer.eos_token_id:
                    break
                test_tokenized_input['input_ids'] = pred_token_id.reshape(1,-1)
                test_tokenized_input['attention_mask'] = None
            pred_str = tokenizer.decode(pred_seq).strip()
            pred_str = pred_str.replace(tokenizer.eos_token, "")
            pred_str = pred_str.split("\n\n")[0]
            print(pred_str)
            print(test_target)
            print("____________")
            if pred_str == test_target:
                correct_cnt+=1
            if pred_seq[0]==target_seq[0]:
                first_token_match+=1
            fv_res.append({"prompt": test_prompt, "gt": test_target, "pred": pred_str, "probs": prob_seq})
        interv_acc = correct_cnt/len(test_dataset)
        FTM = first_token_match/len(test_dataset)
        print(interv_acc)
        print(first_token_match)
        res = {"acc": interv_acc, "res": fv_res}
    with open(os.path.join(res_dir, "fv_result.json"), "w") as f:
        json.dump(res, f)