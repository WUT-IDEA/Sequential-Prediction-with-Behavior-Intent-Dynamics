dataset_name="toys"
max_seq_length=100
masked_lm_prob=0.2
max_predictions_per_seq=40

dim=64
batch_size=256
num_train_steps=85000

prop_sliding_window=0.5
mask_prob=1.0
dupe_factor=10
pool_size=10
alpha=0.5
beta=0
gamma=1
peers=2
manner="min"

signature_para="manner${manner}_peers${peers}_alpha${alpha}-beta${beta}-gamma${gamma}-mp${mask_prob}-sw${prop_sliding_window}-mlp${masked_lm_prob}-df${dupe_factor}-mpps${max_predictions_per_seq}-msl${max_seq_length}"
signature_data="_mp${mask_prob}-sw${prop_sliding_window}-mlp${masked_lm_prob}-df${dupe_factor}-mpps${max_predictions_per_seq}-msl${max_seq_length}"

#python -u gen_data_fin.py \
#    --dataset_name=${dataset_name} \
#    --max_seq_length=${max_seq_length} \
#    --max_predictions_per_seq=${max_predictions_per_seq} \
#    --mask_prob=${mask_prob} \
#    --dupe_factor=${dupe_factor} \
#    --masked_lm_prob=${masked_lm_prob} \
#    --prop_sliding_window=${prop_sliding_window} \
#    --signature=${signature_data} \
#    --pool_size=${pool_size} \

python -u run.py \
    --train_input_file=./data/${dataset_name}${signature_data}.train.tfrecord \
    --test_input_file=./data/${dataset_name}${signature_data}.test.tfrecord \
    --vocab_filename=./data/${dataset_name}${signature_data}.vocab \
    --user_history_filename=./data/${dataset_name}${signature_data}.his \
    --checkpointDir=./CKPT_DIR/${dataset_name}/ \
    --signature=${signature_para}-${dim} \
    --do_train=True \
    --do_eval=True \
    --bert_config_file=./bert_train/bert_config_${dataset_name}_${dim}.json \
    --batch_size=${batch_size} \
    --max_seq_length=${max_seq_length} \
    --max_predictions_per_seq=${max_predictions_per_seq} \
    --num_train_steps=${num_train_steps} \
    --num_warmup_steps=100 \
    --learning_rate=1e-4 \
    --peers=${peers} \
    --manner=${manner} \
    --alpha=${alpha} \
    --beta=${beta} \
    --gamma=${gamma} \

