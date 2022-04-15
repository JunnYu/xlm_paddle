# train
python train.py --output_dir facebook_xnli --pretrained_model_name_or_path xlm-mlm-tlm-xnli15-1024

# eval
python eval.py --output_dir eval_output --pretrained_model_name_or_path xlm-mlm-tlm-xnli15-1024-fintuned-on-xnli