CUDA_VISIBLE_DEVICES=2 python3 act_converter_generation_task.py --dataset_name StylePTB_ATP
CUDA_VISIBLE_DEVICES=2 python3 few_shot_generation.py --dataset_name StylePTB_ATP
CUDA_VISIBLE_DEVICES=2 python3 act_converter_generation_task.py --dataset_name StylePTB_IAD
CUDA_VISIBLE_DEVICES=2 python3 few_shot_generation.py --dataset_name StylePTB_IAD 
CUDA_VISIBLE_DEVICES=2 python3 act_converter_generation_task.py --dataset_name StylePTB_TFU
CUDA_VISIBLE_DEVICES=2 python3 few_shot_generation.py --dataset_name StylePTB_TFU



 