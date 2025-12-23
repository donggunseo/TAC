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
from comet import load_from_checkpoint, download_model
from transformers import get_cosine_schedule_with_warmup
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
    parser.add_argument("--num_shots", type=int, default=10)
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--result_folder", type=str, default='./final_result')
    parser.add_argument("--data_dir", type=str, default='./dataset')
    parser.add_argument("--dataset_name", type=str, default='wmt19_cs-en')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--max_generate_length", type=int, default=30)

    args = parser.parse_args()

    set_seed(args.seed)
    model_card = MODEL_CARD[args.model_name]

    res_dir = os.path.join(args.result_folder, args.dataset_name, args.model_name, str(args.seed))

    os.makedirs(res_dir, exist_ok=True)

    train_dataset, valid_dataset, test_dataset, _ = load_data(args.dataset_name, args.data_dir)

    target_lang_code = args.dataset_name[-2:] if "wmt" in args.dataset_name else 'en'
    
    device = 'cuda'
    model, tokenizer, model_config = load_model_and_tokenizer(model_card)
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
        activation_storage = []
        for _ in range(args.n_trials):
            dummy_query = random.choice(train_dataset)
            demon_pool = [item for item in train_dataset if item!=dummy_query]
            dummy_prompt, dummy_query_target = create_prompt_generation(demon_pool=demon_pool, query = dummy_query, num_shots = args.num_shots)
            dummy_tokenized_input = tokenizer(dummy_prompt, return_tensors='pt').to(device)
            with torch.no_grad() and TraceDict(model, layers=model_config['attn_hook_names'], retain_input=True, retain_output=False) as td:
                outputs = model.forward(**dummy_tokenized_input)
            stack_initial = torch.vstack([split_activations_by_head(td[layer].input, model_config) for layer in model_config['attn_hook_names']]).permute(0,2,1,3)
            cur_activation = stack_initial[:, :, -1, :].cpu().detach()
            activation_storage.append(cur_activation)
        activation_storage = torch.stack(activation_storage)
        activation_storage = torch.mean(activation_storage, dim=0)
        torch.save(activation_storage, os.path.join(res_dir, "head_act.pt"))
    activation_storage.requires_grad=False
    

    def intervention_fn_train(head_act, model_config, batch_size, act_converter, device, batched_convert_idx):
        def mix_head_act(output, layer_name, inputs):
            current_layer = int(layer_name.split(".")[2])           
            intervention_embedding = head_act[current_layer,:,:].unsqueeze(0).to(device)  #shape: (1,n_heads, head_dim)
            if isinstance(inputs, tuple):
                inputs = inputs[0]
            original_shape = inputs.shape # (batch_size, tokens, n_heads*head_dim) 
            new_shape = inputs.size()[:-1] + (model_config['n_heads'], model_config['resid_dim']//model_config['n_heads'])
            inputs = inputs.view(*new_shape) # inputs shape: (batch_size , tokens (n), n_heads, head_dim)
            modified_inputs = inputs.clone()
            
            completion_activations = []
            completion_lengths = []
            for batch_idx in range(batch_size):
                single_input = inputs[batch_idx] # (tokens (n), n_heads, head_dim)
                convert_idx = batched_convert_idx[batch_idx]
                single_input = single_input[convert_idx:] # (generated_tokens, n_heads, head_dim)
                completion_activations.append(single_input)
                completion_lengths.append(single_input.shape[0])
            longest = max(completion_lengths)
            for batch_idx in range(batch_size):
                if completion_activations[batch_idx].shape[0]<longest:
                    completion_activations[batch_idx] = torch.cat([completion_activations[batch_idx], torch.zeros((longest-completion_activations[batch_idx].shape[0], model_config['n_heads'], model_config['resid_dim']//model_config['n_heads']), dtype = torch.bfloat16).to(device)], dim=0)
            completion_activations = torch.stack(completion_activations, dim=0).permute(0,2,1,3).reshape(-1, longest, model_config['resid_dim']//model_config['n_heads']) ## (batch_size, padded_generated_tokens, n_heads, head_dim) -> (batch_size * n_heads, padded_generated_tokens, head_dim)
            repeated_intervention_embedding = intervention_embedding.repeat(1,batch_size,1) 
            out, _ = act_converter(completion_activations, (torch.zeros(*repeated_intervention_embedding.shape, dtype=torch.bfloat16).to(device), repeated_intervention_embedding)) # out shape : (batch_size * n_heads, padded_generated_tokens, head_dim)
            modified = out.reshape(batch_size,model_config['n_heads'],longest,model_config['resid_dim']//model_config['n_heads']).permute(0,2,1,3) ## (batch_size, padded_generated_tokens, n_heads, head_dim)            for batch_idx in range(batch_size):
            for batch_idx in range(batch_size):
                modified_inputs[batch_idx, batched_convert_idx[batch_idx]:] = modified[batch_idx, :completion_lengths[batch_idx]].to(dtype=modified_inputs.dtype)
            modified_inputs = modified_inputs.contiguous()
            modified_inputs = modified_inputs.view(*original_shape)
            proj_module = get_module(model, layer_name)
            out_proj = proj_module.weight
            new_output = torch.matmul(modified_inputs, out_proj.T)
            return new_output
        return mix_head_act

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

    best_val_acc = 0
    best_converter_state = None

    model_path = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(model_path)

    if os.path.exists(os.path.join(res_dir, "best_converter_state.pt")):
        best_converter_state = torch.load(os.path.join(res_dir, "best_converter_state.pt"))
    else:
        total_steps = ((len(train_dataset)+args.batch_size-1)//args.batch_size)*args.epoch
        warmup_steps = int(0.2*total_steps)
        optimizer = torch.optim.AdamW(act_converter.parameters(), lr=args.lr, weight_decay=0.01)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps = warmup_steps, num_training_steps=total_steps)
        early_stop_cnt = 0
        early_stop_ths = 2
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
                    batched_convert_idx = []
                    for train_item in train_items:
                        train_prompt, train_target = create_prompt_generation(demon_pool=None, query = train_item, num_shots=0)
                        batched_train_prompt.append(train_prompt+" "+train_target + "\n\n")
                        target_seq = " "+ train_target+ "\n\n" + tokenizer.eos_token
                        prompt_length = len(tokenizer.encode(train_prompt))
                        target_length = len(tokenizer.encode(target_seq, add_special_tokens = False))
                        batched_train_target.append([-100]*(prompt_length-1) + tokenizer.encode(target_seq, add_special_tokens=False))
                        batched_convert_idx.append(prompt_length-1)
                    train_tokenized_input = tokenizer(batched_train_prompt, return_tensors='pt', padding='longest')
                    train_tokenized_input.to(device)
                    batch_len = len(train_tokenized_input.input_ids[0])
                    for i, b in enumerate(batched_train_target):
                        if len(b)<batch_len:
                            batched_train_target[i]+=[-100]*(batch_len-len(b))
                    batched_train_target = torch.tensor(batched_train_target)
                    intervention_fn = intervention_fn_train(head_act = activation_storage, model_config = model_config, batch_size=cur_batch_size, act_converter=act_converter, device=device, batched_convert_idx=batched_convert_idx)
                    with TraceDict(model, layers=model_config['attn_hook_names'], edit_output=intervention_fn):              
                        output = model.forward(**train_tokenized_input)
                    out_logit = output.logits[torch.arange(cur_batch_size)]
                    loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index = -100)
                    task_loss = loss_fct(out_logit.reshape(-1,out_logit.shape[-1]), batched_train_target.reshape(-1).to(device)).mean()
                    task_loss.backward()
                    torch.nn.utils.clip_grad_norm_(act_converter.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    if idx%10==0:
                        print("train loss : ", task_loss.item())
            torch.cuda.empty_cache()
            with torch.no_grad():
                predictions = []
                for val_item in tqdm(valid_dataset):
                    val_prompt, val_target = create_prompt_generation(demon_pool=None, query = val_item, num_shots=0)
                    val_tokenized_input = tokenizer(val_prompt, return_tensors='pt').to(device)
                    kv_cache = None
                    pred_seq = []
                    act_input = activation_storage
                    for t in range(args.max_generate_length):
                        intervention_fn = intervention_fn_inference(head_act = act_input, model_config = model_config, act_converter=act_converter, device = device, step=t)
                        with TraceDict(model, layers=model_config['attn_hook_names'], edit_output=intervention_fn):              
                            output = model.forward(**val_tokenized_input, use_cache=True, past_key_values = kv_cache)
                            kv_cache = output.past_key_values
                        output_logits = output.logits[0,-1]
                        pred_token_id = torch.argmax(output_logits, dim=-1)
                        pred_seq.append(pred_token_id.item())
                        if pred_token_id==tokenizer.eos_token_id:
                            break
                        val_tokenized_input['input_ids'] = pred_token_id.reshape(1,-1)
                        val_tokenized_input['attention_mask'] = None
                        act_input = h_status.copy()
                        h_status = []
                    pred_str = tokenizer.decode(pred_seq).strip()
                    pred_str = pred_str.replace(tokenizer.eos_token, "")
                    pred_str = pred_str.split("\n\n")[0]
                    print(pred_str)
                    print(val_target)
                    print("_____________")
                    predictions.append({"src" : val_item["input"], "mt": pred_str, "ref" : val_target})
                    h_status = []
                scores = comet_model.predict(predictions, batch_size=16, gpus=1)
                val_acc = sum(scores["scores"])/len(scores["scores"])
                print(f"Epoch {epoch+1} validation comet score : {val_acc}")
                if val_acc>best_val_acc:
                    best_val_acc = val_acc
                    best_converter_state = act_converter.state_dict()
                else:
                    early_stop_cnt+=1
                if early_stop_cnt==early_stop_ths:
                    break
        torch.save(best_converter_state, os.path.join(res_dir, "best_converter_state.pt"))
    act_converter.load_state_dict(best_converter_state)
    act_converter.eval()
    
    softmax = nn.Softmax(dim=-1)
    act_convert_res = []
    new_line_id = tokenizer.encode("\n\n", add_special_tokens=False)[0]
    with torch.no_grad():
        predictions = []
        for test_item in tqdm(test_dataset):
            test_prompt, test_target = create_prompt_generation(demon_pool=None, query = test_item, num_shots=0)
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
                if pred_token_id.item()==new_line_id or pred_token_id.item()==tokenizer.eos_token_id:
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
            predictions.append({"src" : test_item["input"], "mt": pred_str, "ref" : test_target})
            act_convert_res.append({"prompt": test_prompt, "query": test_item["input"], "gt": test_target, "pred": pred_str})
            h_status = []
        scores = comet_model.predict(predictions, batch_size=16, gpus=1)
        interv_acc = sum(scores["scores"])/len(scores["scores"])
        site = {"acc": interv_acc, "res": act_convert_res}    
    with open(os.path.join(res_dir, "extended_act_converter_result.json"), "w") as f:
        json.dump(site, f)