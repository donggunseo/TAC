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

GN_CNT=0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='meta-llama/Llama-3.1-8B')
    parser.add_argument("--num_shots_by_class", type=int, default=2)
    parser.add_argument("--result_folder", type=str, default='./result')
    parser.add_argument("--data_dir", type=str, default='./dataset')
    parser.add_argument("--dataset_name", type=str, default='banking77_3token')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--demon_selection", type=str, default = "stratify")
    parser.add_argument("--lr", type=float, default=2e-1)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=10)
    
    
    args = parser.parse_args()

    set_seed(args.seed)
    model_card = "llama3.1-8b" if "llama" in args.model_name else "qwen2.5-7b"

    res_dir = os.path.join(args.result_folder, args.dataset_name, model_card)

    os.makedirs(res_dir, exist_ok=True)

    train_dataset, valid_dataset, test_dataset, label_list = load_data(args.dataset_name, args.data_dir)
    
    device = 'cuda'
    model, tokenizer, model_config = load_model_and_tokenizer(args.model_name)
    for param in model.parameters():
        param.requires_grad = False
    model.to(device)

    

    #pure ICL
    correct_count = 0
    pure_ICL_res = []
    for test_item in tqdm(test_dataset):
        test_prompt, test_query_target = create_prompt(demon_pool=train_dataset, query = test_item, num_shots_by_class=args.num_shots_by_class, option=args.demon_selection, label_list=label_list, shuffle_label=False)
        test_tokenized_input = tokenizer(test_prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model.generate(**test_tokenized_input, max_new_tokens=3, pad_token_id=tokenizer.eos_token_id, eos_token_id = tokenizer.eos_token_id, tokenizer=tokenizer, do_sample=False, temperature = None, top_p = None)
        output_str = tokenizer.decode(outputs.squeeze()[test_tokenized_input.input_ids[0].shape[0]:], skip_speical_tokens=True)
        if output_str.strip()==test_query_target:
            correct_count+=1
        pure_ICL_res.append({"prompt" : test_prompt, "gt": test_query_target, "pred" : output_str.strip()})
    ICL_acc = correct_count/len(test_dataset)
    pure_ICL = {"acc": ICL_acc, "res": pure_ICL_res}
    with open(os.path.join(res_dir, "pure_ICL_result.json"), "w") as f:
        json.dump(pure_ICL, f)
    
    ##Task vector
    ### activation extraction
    #### 1. random
    # if os.path.exists(os.path.join(res_dir, "tv_act2.pt")):
    #     activation_storage = torch.load(os.path.join(res_dir, "tv_act2.pt"))
    # else:
    #     n_trials = 100
    #     activation_storage = []
    #     for n in range(n_trials):
    #         dummy_query = random.choice(train_dataset)
    #         demon_pool = [item for item in train_dataset if item!=dummy_query]
    #         dummy_prompt, dummy_query_target = create_prompt(demon_pool=demon_pool, query = dummy_query, num_shots_by_class=args.num_shots_by_class, option=args.demon_selection, label_list=label_list, shuffle_label=False)
    #         dummy_tokenized_input = tokenizer(dummy_prompt, return_tensors='pt').to(device)
    #         activation = [None,None,None]
    #         for t in range(3):
    #             with torch.no_grad() and TraceDict(model, layers=model_config['layer_hook_names'], retain_input=False, retain_output=True) as td:                
    #                 outputs = model.forward(**dummy_tokenized_input)
    #             output_logits = outputs.logits[0,-1]
    #             pred_token_id = torch.argmax(output_logits, dim=-1)
    #             activation[t] = torch.stack([td[layer].output[0,-1].cpu().detach() for layer in model_config['layer_hook_names']])
    #             dummy_tokenized_input['input_ids'] = torch.cat((dummy_tokenized_input['input_ids'], pred_token_id.reshape(1,-1)), dim=-1)
    #             dummy_tokenized_input['attention_mask'] = torch.cat((dummy_tokenized_input['attention_mask'], torch.tensor([[1]]).to(device)), dim=-1)
    #         activation = torch.stack(activation)
    #         activation_storage.append(activation)
    #     activation_storage = torch.stack(activation_storage)
    #     activation_storage = torch.mean(activation_storage, dim=0)
    #     torch.save(activation_storage, os.path.join(res_dir, "tv_act2.pt"))



    #### 2. stratify prediction & correct
    # if os.path.exists(os.path.join(res_dir, "tv_act.pt")):
    #     activation_storage = torch.load(os.path.join(res_dir, "tv_act.pt"))
    # else:
    #     train_buckets = defaultdict(list)
    #     for item in train_dataset:
    #         if item["output"] in label_list:
    #             train_buckets[item["output"]].append(item)
    #     n_trials_by_class = 3
    #     activation_storage = []
    #     for l in tqdm(label_list):
    #         dummy_query_cand = train_buckets[l]
    #         act_cnt = 0
    #         while act_cnt!=n_trials_by_class:
    #             dummy_query = random.choice(dummy_query_cand)
    #             demon_pool = [item for item in train_dataset if item!=dummy_query]
    #             dummy_prompt, dummy_query_target = create_prompt(demon_pool=demon_pool, query = dummy_query, num_shots_by_class=args.num_shots_by_class, option=args.demon_selection, label_list=label_list, shuffle_label=False)
    #             dummy_tokenized_input = tokenizer(dummy_prompt, return_tensors='pt').to(device)
    #             activation = [None,None,None]
    #             pred_seq = []
    #             for t in range(3):
    #                 with torch.no_grad() and TraceDict(model, layers=model_config['layer_hook_names'], retain_input=False, retain_output=True) as td:                
    #                     outputs = model.forward(**dummy_tokenized_input)
    #                 output_logits = outputs.logits[0,-1]
    #                 pred_token_id = torch.argmax(output_logits, dim=-1)
    #                 pred_seq.append(pred_token_id.item())
    #                 activation[t] = torch.stack([td[layer].output[0,-1].cpu().detach() for layer in model_config['layer_hook_names']])
    #                 dummy_tokenized_input['input_ids'] = torch.cat((dummy_tokenized_input['input_ids'], pred_token_id.reshape(1,-1)), dim=-1)
    #                 dummy_tokenized_input['attention_mask'] = torch.cat((dummy_tokenized_input['attention_mask'], torch.tensor([[1]]).to(device)), dim=-1)
    #             pred_str = tokenizer.decode(pred_seq).strip()
    #             if pred_str == dummy_query_target:
    #                 act_cnt+=1
    #                 activation = torch.stack(activation)
    #                 activation_storage.append(activation)
    #             else:
    #                 continue
    #     activation_storage = torch.stack(activation_storage)
    #     activation_storage = torch.mean(activation_storage, dim=0)
    #     torch.save(activation_storage, os.path.join(res_dir, "tv_act.pt"))


    # def intervention_fn_tv(edit_layer, tv_activation, device, step_idx):
    #     def replace_activation(output, layer_name, inputs):
    #         current_layer = int(layer_name.split(".")[2])
    #         if current_layer==edit_layer:
    #             output[0, -1] = tv_activation[step_idx][current_layer].to(device)
    #         else:
    #             return output
    #     return replace_activation

    # softmax = nn.Softmax(dim=-1)
    
    ### best_layer_check
    # layer_val_score = {l:0 for l in range(model_config["n_layers"])}
    # for l in tqdm(range(model_config["n_layers"])):
    #     correct_cnt = 0
    #     for valid_item in valid_dataset:
    #         test_prompt, test_query_target = create_prompt(demon_pool=None, query = valid_item, num_shots_by_class=0, option=None, label_list=label_list, shuffle_label=False)
    #         test_tokenized_input = tokenizer(test_prompt, return_tensors='pt').to(device)
    #         pred_seq = []
    #         kv_cache = None
    #         for t in range(3):
    #             intervention_fn = intervention_fn_tv(edit_layer= l, tv_activation=activation_storage, device=device, step_idx = t)
    #             with torch.no_grad() and TraceDict(model, layers = model_config['layer_hook_names'], edit_output=intervention_fn):
    #                 outputs = model.forward(**test_tokenized_input, use_cache=True, past_key_values = kv_cache)
    #                 kv_cache = outputs.past_key_values
    #             output_logits = outputs.logits[0,-1]
    #             output_prob = softmax(output_logits)
    #             values, indices = torch.topk(output_prob, k=10)
    #             pred_token_id = torch.argmax(output_logits, dim=-1)
    #             pred_seq.append(pred_token_id.item())
    #             test_tokenized_input['input_ids'] = torch.cat((test_tokenized_input['input_ids'], pred_token_id.reshape(1,-1)), dim=-1)
    #             test_tokenized_input['attention_mask'] = torch.cat((test_tokenized_input['attention_mask'], torch.tensor([[1]]).to(device)), dim=-1)
    #         pred_str = tokenizer.decode(pred_seq).strip()
    #         if pred_str == test_query_target:
    #             correct_cnt+=1
    #     layer_val_score[l] = correct_cnt/len(valid_dataset)
    # print(layer_val_score)
        
        
    ##SITE

    ### head_activation extraction
    def split_activations_by_head(activations, model_config):
        new_shape = activations.size()[:-1] + (model_config['n_heads'], model_config['resid_dim']//model_config['n_heads']) # split by head: + (n_attn_heads, hidden_size/n_attn_heads)
        activations = activations.view(*new_shape)  # (batch_size, n_tokens, n_heads, head_hidden_dim)
        return activations
    
    if os.path.exists(os.path.join(res_dir, "head_act.pt")):
        activation_storage = torch.load(os.path.join(res_dir, "head_act.pt"))
        print(activation_storage.shape)
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
                activation = [None,None,None]
                pred_seq = []
                for t in range(3):
                    with torch.no_grad() and TraceDict(model, layers=model_config['attn_hook_names'], retain_input=True, retain_output=False) as td:                
                        outputs = model.forward(**dummy_tokenized_input)
                    output_logits = outputs.logits[0,-1]
                    pred_token_id = torch.argmax(output_logits, dim=-1)
                    pred_seq.append(pred_token_id.item())
                    stack_initial = torch.vstack([split_activations_by_head(td[layer].input, model_config) for layer in model_config['attn_hook_names']]).permute(0,2,1,3)
                    cur_activation = stack_initial[:, :, -1, :].cpu().detach()
                    activation[t] = cur_activation
                    dummy_tokenized_input['input_ids'] = torch.cat((dummy_tokenized_input['input_ids'], pred_token_id.reshape(1,-1)), dim=-1)
                    dummy_tokenized_input['attention_mask'] = torch.cat((dummy_tokenized_input['attention_mask'], torch.tensor([[1]]).to(device)), dim=-1)
                pred_str = tokenizer.decode(pred_seq).strip()
                if pred_str == dummy_query_target:
                    act_cnt+=1
                    activation = torch.stack(activation)
                    activation_storage.append(activation)
                else:
                    continue
        activation_storage = torch.stack(activation_storage)
        activation_storage = torch.mean(activation_storage, dim=0)
        print(activation_storage.shape)
        torch.save(activation_storage, os.path.join(res_dir, "head_act.pt"))

    ### alpha optimization
    def alpha_intervention_fn(alphas, head_act, model_config, batch_size, device):
        def mix_head_act(output, layer_name, inputs):
            current_layer = int(layer_name.split(".")[2])
            intervention_embedding = head_act[:, current_layer,:,:].unsqueeze(0).to(device)
            if isinstance(inputs, tuple):
                inputs = inputs[0]
            original_shape = inputs.shape
            new_shape = inputs.size()[:-1] + (model_config['n_heads'], model_config['resid_dim']//model_config['n_heads'])
            inputs = inputs.view(*new_shape) # inputs shape: (batch_size , tokens (n), n_heads, head_dim)

            modified_inputs = inputs.clone()
            gated_value = alphas[:, current_layer,:].unsqueeze(0).unsqueeze(-1).to(device)

            modified = (1 - gated_value) * inputs[torch.arange(batch_size), -3:, :, :] + gated_value * intervention_embedding
            modified_inputs[torch.arange(batch_size), -3:] = modified.to(dtype=modified_inputs.dtype) # (batch_size, n_heads, head_dim)
            modified_inputs = modified_inputs.contiguous()

            modified_inputs = modified_inputs.view(*original_shape)

            proj_module = get_module(model, layer_name)

            out_proj = proj_module.weight
            new_output = torch.matmul(modified_inputs, out_proj.T)

            return new_output
        return mix_head_act


    
    eps = 1e-3

    best_val_loss = float('inf')
    best_alphas_logit = None

    alpha_shape = (3, model_config["n_layers"], model_config["n_heads"])
    init_logit = 0.0 
    alphas_logit = torch.nn.Parameter(torch.full(alpha_shape, init_logit, device='cuda'))
    optimizer = torch.optim.Adam([alphas_logit], lr=args.lr)

    tokenizer.padding_side = "left"

    if os.path.exists(os.path.join(res_dir, "alpha.pt")):
        best_alphas_logit = torch.load(os.path.join(res_dir, "alpha.pt"))
        print(best_alphas_logit.shape)
    else:
        with torch.set_grad_enabled(True):
            for epoch in tqdm(range(args.epoch)):
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
                        batched_train_prompt.append(train_prompt + " " + train_target)
                        batched_train_target.append(train_target)
                    train_tokenized_input = tokenizer(batched_train_prompt, return_tensors='pt', padding='longest').to(device)
                    batched_target_tokens = torch.tensor([tokenizer.encode(": "+t, add_special_tokens=False)[1:] for t in batched_train_target]).to(device)


                    alphas = torch.sigmoid(alphas_logit)
                    intervention_fn = alpha_intervention_fn(alphas = alphas, head_act = activation_storage, model_config = model_config, batch_size=cur_batch_size, device=device)
                    with TraceDict(model, layers=model_config['attn_hook_names'], edit_output=intervention_fn, retain_grad=True) as td:              
                        output = model.forward(input_ids = train_tokenized_input.input_ids[:,:-1], attention_mask= train_tokenized_input.attention_mask[:,:-1])
                    out_logit = output.logits[torch.arange(cur_batch_size),-3:,:]
                    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                    task_loss = loss_fct(out_logit.reshape(-1,out_logit.shape[-1]), train_tokenized_input.input_ids[:,-3:].reshape(-1)).mean()
                    task_loss.backward()
                    optimizer.step()
                    if idx%10==0:
                        print(f"train loss : {task_loss.item()} with alpha {alphas.mean().item():.4f}")
                torch.cuda.empty_cache()
                with torch.no_grad():
                    loss_list = []
                    for idx, batch_start in tqdm(enumerate(range(0, len(valid_dataset), args.batch_size))):
                        optimizer.zero_grad()
                        batch_end = min(batch_start + args.batch_size, len(valid_dataset))
                        cur_batch_size = batch_end - batch_start
                        valid_items = [valid_dataset[i] for i in range(batch_start, batch_end)]
                        batched_valid_prompt = []
                        batched_valid_target = []
                        for valid_item in valid_items:
                            valid_prompt, valid_target = create_prompt(demon_pool=None, query = valid_item, num_shots_by_class=0, option=None, label_list=label_list, shuffle_label=False)
                            batched_valid_prompt.append(valid_prompt + " " + valid_target)
                            batched_valid_target.append(valid_target)
                        valid_tokenized_input = tokenizer(batched_valid_prompt, return_tensors='pt', padding='longest').to(device)
                        batched_target_tokens = torch.tensor([tokenizer.encode(": "+t, add_special_tokens=False)[1:] for t in batched_valid_target]).to(device)


                        alphas = torch.sigmoid(alphas_logit)
                        intervention_fn = alpha_intervention_fn(alphas = alphas, head_act = activation_storage, model_config = model_config, batch_size=cur_batch_size, device=device)
                        with TraceDict(model, layers=model_config['attn_hook_names'], edit_output=intervention_fn, retain_grad=True) as td:              
                            output = model.forward(input_ids = valid_tokenized_input.input_ids[:,:-1], attention_mask= valid_tokenized_input.attention_mask[:,:-1])
                        out_logit = output.logits[torch.arange(cur_batch_size),-3:,:]
                        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                        task_loss = loss_fct(out_logit.reshape(-1,out_logit.shape[-1]), valid_tokenized_input.input_ids[:,-3:].reshape(-1)).mean()
                        loss_list.append(task_loss.item())
                    val_loss = sum(loss_list)/len(loss_list)
                    print(f"Epoch {epoch} validation loss : {val_loss}")
                    if val_loss<best_val_loss:
                        best_val_loss = val_loss
                        best_alphas_logit = alphas_logit.detach().clone()
        torch.save(best_alphas_logit, os.path.join(res_dir, "alpha.pt"))
    
    
    def alpha_intervention_fn_inference(alphas, head_act, model_config, batch_size, device):
        def mix_head_act_inference(output, layer_name, inputs):
            current_layer = int(layer_name.split(".")[2])
            intervention_embedding = head_act[current_layer,:,:].unsqueeze(0).unsqueeze(1).to(device)
            if isinstance(inputs, tuple):
                inputs = inputs[0]
            original_shape = inputs.shape
            new_shape = inputs.size()[:-1] + (model_config['n_heads'], model_config['resid_dim']//model_config['n_heads'])
            inputs = inputs.view(*new_shape) # inputs shape: (batch_size , tokens (n), n_heads, head_dim)

            modified_inputs = inputs.clone()
            gated_value = alphas[current_layer,:].reshape(1,1,-1,1).to(device)

            modified = (1 - gated_value) * inputs[torch.arange(batch_size), -1, :, :].unsqueeze(1) + gated_value * intervention_embedding
            modified_inputs[torch.arange(batch_size), -1] = modified.squeeze(1).to(dtype=modified_inputs.dtype) # (batch_size, n_heads, head_dim)
            modified_inputs = modified_inputs.contiguous()

            modified_inputs = modified_inputs.view(*original_shape)

            proj_module = get_module(model, layer_name)

            out_proj = proj_module.weight
            new_output = torch.matmul(modified_inputs, out_proj.T)

            return new_output
        return mix_head_act_inference
    ## test

    softmax = nn.Softmax(dim=-1)
    correct_cnt = 0
    SITE_res = []
    with torch.no_grad():
        for test_item in tqdm(test_dataset):
            test_prompt, test_target = create_prompt(demon_pool=None, query = test_item, num_shots_by_class=0, option=None, label_list=label_list, shuffle_label=False)
            test_tokenized_input = tokenizer(test_prompt, return_tensors='pt').to(device)
            alphas = torch.sigmoid(best_alphas_logit)
            kv_cache = None
            pred_seq = []
            for t in range(3):
                intervention_fn = alpha_intervention_fn_inference(alphas = alphas[t], head_act = activation_storage[t], model_config = model_config, batch_size=1, device = device)
                with TraceDict(model, layers=model_config['attn_hook_names'], edit_output=intervention_fn) as td:              
                    output = model.forward(**test_tokenized_input, use_cache=True, past_key_values = kv_cache)
                    kv_cache = output.past_key_values
                output_logits = output.logits[0,-1]
                output_prob = softmax(output_logits)
                pred_token_id = torch.argmax(output_logits, dim=-1)
                pred_seq.append(pred_token_id.item())
                test_tokenized_input['input_ids'] = pred_token_id.reshape(1,-1)
                test_tokenized_input['attention_mask'] = None
            pred_str = tokenizer.decode(pred_seq).strip()
            if pred_str == test_target:
                correct_cnt+=1
            SITE_res.append({"prompt": test_prompt, "gt": test_target, "pred": pred_str})
        interv_acc = correct_cnt/len(test_dataset)
        print(interv_acc)
        site = {"acc": interv_acc, "res": SITE_res}
    with open(os.path.join(res_dir, "SITE_result.json"), "w") as f:
        json.dump(site, f)
                
                