CUDA_VISIBLE_DEVICES=0 python3 LoRA_tuning_cls.py --dataset_name banking77
CUDA_VISIBLE_DEVICES=0 python3 LoRA_tuning_cls.py --dataset_name trec_fine
CUDA_VISIBLE_DEVICES=0 python3 LoRA_tuning_cls.py --dataset_name clinc150

# CUDA_VISIBLE_DEVICES=0 python3 LoRA_tuning_cls.py --dataset_name banking77 --model_name qwen2.5-7b
# CUDA_VISIBLE_DEVICES=0 python3 LoRA_tuning_cls.py --dataset_name trec_fine --model_name qwen2.5-7b
# CUDA_VISIBLE_DEVICES=0 python3 LoRA_tuning_cls.py --dataset_name clinc150 --model_name qwen2.5-7b

# CUDA_VISIBLE_DEVICES=0 python3 LoRA_tuning_cls.py --dataset_name banking77 --model_name llama3.2-1b
# CUDA_VISIBLE_DEVICES=0 python3 LoRA_tuning_cls.py --dataset_name trec_fine --model_name llama3.2-1b
# CUDA_VISIBLE_DEVICES=0 python3 LoRA_tuning_cls.py --dataset_name clinc150 --model_name llama3.2-1b

# CUDA_VISIBLE_DEVICES=0 python3 LoRA_tuning_cls.py --dataset_name banking77 --model_name qwen2.5-1.5b
# CUDA_VISIBLE_DEVICES=0 python3 LoRA_tuning_cls.py --dataset_name trec_fine --model_name qwen2.5-1.5b
# CUDA_VISIBLE_DEVICES=0 python3 LoRA_tuning_cls.py --dataset_name clinc150 --model_name qwen2.5-1.5b