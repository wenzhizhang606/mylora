#评测
分为两种模式，评测下游任务


#mylora-params

CUDA_VISIBLE_DEVICES=0,5  python run_crispedit.py  --model llama3-8b --data_type zsre  --energy_threshold 0.5    --batch_size 32  --projection_method_lora v2_grad --alg_name mylora