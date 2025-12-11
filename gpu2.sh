CUDA_VISIBLE_DEVICES=2 python3 act_converter_generation_task.py --dataset_name StylePTB_ATP --model_name Qwen/Qwen2.5-7B
CUDA_VISIBLE_DEVICES=2 python3 few_shot_generation.py --dataset_name StylePTB_ATP --model_name Qwen/Qwen2.5-7B
CUDA_VISIBLE_DEVICES=2 python3 act_converter_generation_task.py --dataset_name StylePTB_IAD --model_name Qwen/Qwen2.5-7B
CUDA_VISIBLE_DEVICES=2 python3 few_shot_generation.py --dataset_name StylePTB_IAD --model_name Qwen/Qwen2.5-7B
CUDA_VISIBLE_DEVICES=2 python3 act_converter_generation_task.py --dataset_name StylePTB_TFU --model_name Qwen/Qwen2.5-7B
CUDA_VISIBLE_DEVICES=2 python3 few_shot_generation.py --dataset_name StylePTB_TFU --model_name Qwen/Qwen2.5-7B
