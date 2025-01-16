python hp_tuning_no_peft.py \
--model_name indolem/indobertweet-base-uncased \
--output_dir best_params \
--num_trials 10 \
--save_best_params best_params/indobertweet.json

python hp_tuning_no_peft.py \
--model_name FacebookAI/xlm-roberta-large \
--output_dir best_params \
--num_trials 10 \
--save_best_params best_params/xlm-r-large.json

python hp_tuning_no_peft.py \
--model_name FacebookAI/xlm-roberta-base \
--output_dir best_params \
--num_trials 10 \
--save_best_params best_params/xlm-r-base.json

python hp_tuning_no_peft.py \
--model_name google-bert/bert-base-multilingual-cased \
--output_dir best_params \
--num_trials 10 \
--save_best_params best_params/mbert.json