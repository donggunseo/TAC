import argparse
import os
import json
import torch
from tqdm import tqdm
from model_utils import *
from data_utils import *
from prompt_utils import *
torch.set_grad_enabled(False)
from baukit import TraceDict, get_module
import torch.nn as nn

from collections import defaultdict
from act_converter_model import activation_converter

MODEL_CARD={
    "meta-llama/Llama-3.1-8B": "llama3.1-8b",
    "Qwen/Qwen2.5-7B" : "qwen2.5-7b",
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='meta-llama/Llama-3.1-8B')
    parser.add_argument("--num_shots_by_class", type=int, default=2)
    parser.add_argument("--result_folder", type=str, default='./result_extended')
    parser.add_argument("--data_dir", type=str, default='./dataset')
    parser.add_argument("--dataset_name", type=str, default='banking77')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--demon_selection", type=str, default = "stratify")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=10)

    
    args = parser.parse_args()

    set_seed(args.seed)
    model_card = MODEL_CARD[args.model_name]

    res_dir = os.path.join(args.result_folder, args.dataset_name, model_card)

    os.makedirs(res_dir, exist_ok=True)

    train_dataset, valid_dataset, test_dataset, label_list = load_data(args.dataset_name, args.data_dir)
    
    device = 'cuda'
    model, tokenizer, model_config = load_model_and_tokenizer(args.model_name)
    for param in model.parameters():
        param.requires_grad = False
    model.to(device)

    def split_activations_by_head(activations, model_config):
        new_shape = activations.size()[:-1] + (model_config['n_heads'], model_config['resid_dim']//model_config['n_heads']) # split by head: + (n_attn_heads, hidden_size/n_attn_heads)
        activations = activations.view(*new_shape)  # (batch_size, n_tokens, n_heads, head_hidden_dim)
        return activations
    
    if os.path.exists(os.path.join(res_dir, "head_act.pt")):
        activation_storage = torch.load(os.path.join(res_dir, "head_act.pt"))
    else:
        train_buckets = defaultdict(list)
        for item in train_dataset:
            if item["output"] in label_list:
                train_buckets[item["output"]].append(item)
        n_trials_by_class = 3
        activation_storage = []
        for l in tqdm(label_list):
            dummy_query_cand = train_buckets[l]
            act_cnt = 0
            while act_cnt!=n_trials_by_class:
                dummy_query = random.choice(dummy_query_cand)
                demon_pool = [item for item in train_dataset if item!=dummy_query]
                dummy_prompt, dummy_query_target = create_prompt(demon_pool=demon_pool, query = dummy_query, num_shots_by_class=args.num_shots_by_class, option=args.demon_selection, label_list=label_list, shuffle_label=False)
                dummy_tokenized_input = tokenizer(dummy_prompt, return_tensors='pt').to(device)
                outputs = model.generate(**dummy_tokenized_input, max_new_tokens = 10, pad_token_id=tokenizer.eos_token_id, eos_token_id = tokenizer.eos_token_id, tokenizer=tokenizer, do_sample=False, temperature = None, top_p = None)
                output_str = tokenizer.decode(outputs.squeeze()[len(dummy_tokenized_input.input_ids.squeeze()):], skip_speical_tokens=True)
                output_str = output_str.strip()
                output_str = output_str.replace(tokenizer.eos_token, "")
                output_str = output_str.split("\n\n")[0]
                if output_str == dummy_query_target:
                    with torch.no_grad() and TraceDict(model, layers=model_config['attn_hook_names'], retain_input=True, retain_output=False) as td:
                        outputs = model.forward(**dummy_tokenized_input)
                    stack_initial = torch.vstack([split_activations_by_head(td[layer].input, model_config) for layer in model_config['attn_hook_names']]).permute(0,2,1,3)
                    cur_activation = stack_initial[:, :, -1, :].cpu().detach()
                    act_cnt+=1
                    activation_storage.append(cur_activation)
                else:
                    continue
        activation_storage = torch.stack(activation_storage)
        activation_storage = torch.mean(activation_storage, dim=0)
        torch.save(activation_storage, os.path.join(res_dir, "head_act.pt"))

    act_converter = activation_converter(input_size=128, hidden_size=128, batch_first=True)
    act_converter.to(device)
    
    def intervention_fn_train(head_act, model_config, batch_size, act_converter, device, cur_tokens=10):
        def mix_head_act(output, layer_name, inputs):
            current_layer = int(layer_name.split(".")[2])           
            intervention_embedding = head_act[current_layer,:,:].to(device)  #shape: (n_heads, head_dim)
            if isinstance(inputs, tuple):
                inputs = inputs[0]
            original_shape = inputs.shape
            new_shape = inputs.size()[:-1] + (model_config['n_heads'], model_config['resid_dim']//model_config['n_heads'])
            inputs = inputs.view(*new_shape) # inputs shape: (batch_size , tokens (n), n_heads, head_dim)

            modified_inputs = inputs.clone()
            intervention_embedding = intervention_embedding.repeat(batch_size,1).unsqueeze(0).contiguous() #shape: (1, batch_size*n_heads, head_dim)
            inputs = inputs[torch.arange(batch_size), -cur_tokens:, :, :].permute(0,2,1,3) #shape: (batch_size ,n_heads, 3, head_dim)
            inputs = inputs.reshape(-1, cur_tokens, model_config['resid_dim']//model_config['n_heads']).contiguous() #shape: (batch_size * n_heads, 3, head_dim)
            hidden_shape = intervention_embedding.shape
            out, _ = act_converter(inputs, (torch.zeros(*hidden_shape, dtype=torch.bfloat16).to(device), intervention_embedding)) # out shape : (batch_size*n_heads, 3, head_dim)
            new_shape = (batch_size, model_config['n_heads'], cur_tokens, model_config['resid_dim']//model_config['n_heads'])
            modified = out.view(*new_shape).permute(0,2,1,3)
            modified_inputs[torch.arange(batch_size), -cur_tokens:] += modified.to(dtype=modified_inputs.dtype) # (batch_size, n_heads, head_dim)
            modified_inputs = modified_inputs.contiguous()

            modified_inputs = modified_inputs.view(*original_shape)

            proj_module = get_module(model, layer_name)

            out_proj = proj_module.weight
            new_output = torch.matmul(modified_inputs, out_proj.T)

            return new_output
        return mix_head_act

    h_status = []
    def intervention_fn_inference(head_act, model_config, batch_size, act_converter, device, step):
        def mix_head_act_inference(output, layer_name, inputs):
            current_layer = int(layer_name.split(".")[2])
            if step==0:
                intervention_embedding = head_act[current_layer,:,:].to(device)
                intervention_embedding = intervention_embedding.unsqueeze(0).contiguous() #shape: (1, batch_size*n_heads, head_dim)
                hidden_shape = intervention_embedding.shape
                intervention_embedding = (torch.zeros(*hidden_shape, dtype=torch.bfloat16).to(device), intervention_embedding)
            else:
                intervention_embedding = head_act[current_layer]
            if isinstance(inputs, tuple):
                inputs = inputs[0]
            original_shape = inputs.shape
            new_shape = inputs.size()[:-1] + (model_config['n_heads'], model_config['resid_dim']//model_config['n_heads'])
            inputs = inputs.view(*new_shape) # inputs shape: (batch_size , tokens (n), n_heads, head_dim)

            modified_inputs = inputs.clone()
            inputs = inputs[torch.arange(batch_size), -1, :, :].unsqueeze(1).permute(0,2,1,3) #shape: (batch_size ,n_heads, 1, head_dim)
            inputs = inputs.reshape(-1, 1, model_config['resid_dim']//model_config['n_heads']).contiguous() #shape: (batch_size * n_heads, 1, head_dim)
            out, status = act_converter(inputs, intervention_embedding) # out shape : (batch_size*n_heads, 1, head_dim)
            global h_status
            h_status.append(status)
            new_shape = (batch_size, model_config['n_heads'], 1, model_config['resid_dim']//model_config['n_heads'])
            modified = out.view(*new_shape).permute(0,2,1,3)
            modified_inputs[torch.arange(batch_size), -1:] += modified.to(dtype=modified_inputs.dtype) # (batch_size, n_heads, head_dim)

            modified_inputs = modified_inputs.view(*original_shape)
            proj_module = get_module(model, layer_name)

            out_proj = proj_module.weight
            new_output = torch.matmul(modified_inputs, out_proj.T)

            return new_output
        return mix_head_act_inference
    

    best_val_acc = 0
    best_converter_state = None

    optimizer = torch.optim.AdamW(act_converter.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5e-3, steps_per_epoch=(len(train_dataset)//args.batch_size)+1, epochs=10, pct_start=0.1, div_factor=10)


    tokenizer.padding_side = "left"

    if os.path.exists(os.path.join(res_dir, "best_converter_state.pt")):
        best_converter_state = torch.load(os.path.join(res_dir, "best_converter_state.pt"))
    else:
        for epoch in tqdm(range(args.epoch)):
            with torch.set_grad_enabled(True):
                random.shuffle(train_dataset)
                for idx, batch_start in tqdm(enumerate(range(0, len(train_dataset), args.batch_size))):
                    optimizer.zero_grad()
                    batch_end = min(batch_start + args.batch_size, len(train_dataset))
                    cur_batch_size = batch_end - batch_start
                    train_items = [train_dataset[i] for i in range(batch_start, batch_end)]
                    batched_train_prompt = []
                    batched_train_target = []
                    for train_item in train_items:
                        train_prompt, train_target = create_prompt(demon_pool=None, query = train_item, num_shots_by_class=0, option=None, label_list=label_list, shuffle_label=False)
                        batched_train_prompt.append(train_prompt)
                        batched_train_target.append(" "+train_target+"\n\n")
                    train_tokenized_input = tokenizer(batched_train_prompt, return_tensors='pt', padding='longest')
                    batched_target_tokens = tokenizer(batched_train_target, return_tensors='pt', padding='longest', padding_side = 'right')
                    cut_idx = 0 if tokenizer.bos_token_id is None else 1
                    batched_target= batched_target_tokens['input_ids'][:,cut_idx:]
                    cur_target_tokens = batched_target.shape[1]
                    train_tokenized_input['input_ids'] = torch.cat((train_tokenized_input['input_ids'], batched_target_tokens['input_ids'][:,cut_idx:-1]), dim=-1)
                    train_tokenized_input['attention_mask'] = torch.cat((train_tokenized_input['attention_mask'], batched_target_tokens['attention_mask'][:,cut_idx:-1]), dim=-1)
                    train_tokenized_input.to(device)
                    batched_target[batched_target==tokenizer.pad_token_id]=-100
                    intervention_fn = intervention_fn_train(head_act = activation_storage, model_config = model_config, batch_size=cur_batch_size, act_converter=act_converter, device=device, cur_tokens=cur_target_tokens)
                    with TraceDict(model, layers=model_config['attn_hook_names'], edit_output=intervention_fn):              
                        output = model.forward(**train_tokenized_input)
                    out_logit = output.logits[torch.arange(cur_batch_size),-cur_target_tokens:,:]
                    loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index = -100)
                    task_loss = loss_fct(out_logit.reshape(-1,out_logit.shape[-1]), batched_target.reshape(-1).to(device)).mean()
                    task_loss.backward()
                    torch.nn.utils.clip_grad_norm_(act_converter.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    if idx%50==0:
                        print("train loss : ", task_loss.item())
            torch.cuda.empty_cache()
            with torch.no_grad():
                correct_cnt=0
                for val_item in tqdm(valid_dataset):
                    val_prompt, val_target = create_prompt(demon_pool=None, query = val_item, num_shots_by_class=0, option=None, label_list=label_list, shuffle_label=False)
                    val_tokenized_input = tokenizer(val_prompt, return_tensors='pt').to(device)
                    kv_cache = None
                    pred_seq = []
                    for t in range(10):
                        if t==0:
                            act_input = activation_storage
                        else:
                            act_input = h_status
                        intervention_fn = intervention_fn_inference(head_act = act_input, model_config = model_config, batch_size=1, act_converter=act_converter, device = device, step=t)
                        with TraceDict(model, layers=model_config['attn_hook_names'], edit_output=intervention_fn):              
                            output = model.forward(**val_tokenized_input, use_cache=True, past_key_values = kv_cache)
                            kv_cache = output.past_key_values
                        output_logits = output.logits[0,-1]
                        pred_token_id = torch.argmax(output_logits, dim=-1)
                        pred_seq.append(pred_token_id.item())
                        val_tokenized_input['input_ids'] = pred_token_id.reshape(1,-1)
                        val_tokenized_input['attention_mask'] = None
                    pred_str = tokenizer.decode(pred_seq).strip()
                    pred_str = pred_str.split("\n\n")[0]
                    print(pred_str)
                    print(val_target)
                    print("_____________")
                    if pred_str == val_target:
                        correct_cnt+=1
                    h_status = []
                val_acc = correct_cnt/len(valid_dataset)
                print(f"Epoch {epoch} validation acc : {val_acc}")
                if val_acc>=best_val_acc:
                    best_val_acc = val_acc
                    best_converter_state = act_converter.state_dict()
        torch.save(best_converter_state, os.path.join(res_dir, "best_converter_state.pt"))
    act_converter.load_state_dict(best_converter_state)
    act_converter.eval()
    
    softmax = nn.Softmax(dim=-1)
    correct_cnt = 0
    act_convert_res = []
    new_line_id = tokenizer.encode("\n\n", add_special_tokens=False)[0]
    with torch.no_grad():
        for test_item in tqdm(test_dataset):
            test_prompt, test_target = create_prompt(demon_pool=None, query = test_item, num_shots_by_class=0, option=None, label_list=label_list, shuffle_label=False)
            test_tokenized_input = tokenizer(test_prompt, return_tensors='pt').to(device)
            kv_cache = None
            pred_seq = []
            pred_probs = {}
            for t in range(10):
                if t==0:
                    act_input = activation_storage
                else:
                    act_input = h_status
                intervention_fn = intervention_fn_inference(head_act = act_input, model_config = model_config, batch_size=1, act_converter=act_converter, device = device, step=t)
                with TraceDict(model, layers=model_config['attn_hook_names'], edit_output=intervention_fn) as td:              
                    output = model.forward(**test_tokenized_input, use_cache=True, past_key_values = kv_cache)
                    kv_cache = output.past_key_values
                output_logits = output.logits[0,-1]
                output_prob = softmax(output_logits)
                values, indices = torch.topk(output_prob, k=10)
                pred_probs[t]={tokenizer.decode(i.item()):v.item() for v,i in zip(values, indices)}
                pred_token_id = torch.argmax(output_logits, dim=-1)
                pred_seq.append(pred_token_id.item())
                if pred_token_id.item()==new_line_id:
                    break
                test_tokenized_input['input_ids'] = pred_token_id.reshape(1,-1)
                test_tokenized_input['attention_mask'] = None
            pred_str = tokenizer.decode(pred_seq).strip()
            pred_str = pred_str.split("\n\n")[0]
            print(pred_str)
            print(test_target)
            print("____________")
            if pred_str == test_target:
                correct_cnt+=1
            act_convert_res.append({"prompt": test_prompt, "gt": test_target, "pred": pred_str, "prob":pred_probs})
            h_status = []
        interv_acc = correct_cnt/len(test_dataset)
        print(interv_acc)
        site = {"acc": interv_acc, "res": act_convert_res}
    with open(os.path.join(res_dir, "extended_act_converter_result.json"), "w") as f:
        json.dump(site, f)