CUDA_VISIBLE_DEVICES=2 python3 LoRA_tuning.py --dataset_name banking77
CUDA_VISIBLE_DEVICES=2 python3 LoRA_tuning.py --dataset_name trec_fine
CUDA_VISIBLE_DEVICES=2 python3 LoRA_tuning.py --dataset_name clinc150

CUDA_VISIBLE_DEVICES=2 python3 LoRA_tuning.py --dataset_name banking77 --model_name Qwen/Qwen2.5-7B
CUDA_VISIBLE_DEVICES=2 python3 LoRA_tuning.py --dataset_name trec_fine --model_name Qwen/Qwen2.5-7B
CUDA_VISIBLE_DEVICES=2 python3 LoRA_tuning.py --dataset_name clinc150 --model_name Qwen/Qwen2.5-7B