import argparse
from templates import set_template


parser = argparse.ArgumentParser(description='RecPlay')
dataset_items = {'toys': 11924,
                 'nyc16': 440,
                 'chi18': 246,
                 'beauty': 12101}
################
# Test
################
parser.add_argument('--num_negative_elements', type=int, default=100)
parser.add_argument('--selection', choices=['popular', 'random'], type=str, default='popular')
parser.add_argument('--metric_ks', nargs='+', type=int, default=[1, 5, 10], help='ks for Metric@k')
parser.add_argument('--log_test', type=int, default=1) # record the result of test dataset, 0 not record
parser.add_argument('--best_metric', type=str, default='ndcg_5')

################
# Dataloader
################
parser.add_argument('--batch_size', type=int, default=256)#256
parser.add_argument('--dataset_code', type=str, default='ml-1m')#ml-1m, toys, nyc16, chi18
parser.add_argument('--pool_size', type=int, default=10)
parser.add_argument('--max_seq_length', type=int, default=200)
parser.add_argument('--max_predictions_per_seq', type=int, default=20)
parser.add_argument('--masked_lm_prob', type=float, default=0.15)
parser.add_argument('--mask_prob', type=float, default=0.8)
parser.add_argument('--dupe_factor', type=int, default=10)
parser.add_argument('--prop_sliding_window', type=float, default=0.1)
parser.add_argument('--random_shuffle_seed', type=int, default=12345)
parser.add_argument('--density', type=float, default=None)
parser.add_argument('--version_id', type=str, default='bertv1')


################
# Trainer
################
# device #
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
parser.add_argument('--num_gpu', type=int, default=-1)
parser.add_argument('--device_idx', type=str, default='0')
# Learning paramaters #
parser.add_argument('--resume_experiment_dir', type=str, default='', help='Go on training from a checkpoint.')
parser.add_argument('--num_epochs', type=int, default=400, help='Number of epochs for training')

# parser.add_argument('--factor', type=float, default=1, help='The overall scale factor for the learning rate decay.')
parser.add_argument('--init_lr', type=float, default=1e-3, help='The initial learning rate for Adam.')
parser.add_argument('--n_warmup_steps', type=int, default=100, help='Number of warmup steps')
parser.add_argument('--clip_norm', type=float, default=5.0, help='Clip grad norm')

################
# Model
################
parser.add_argument('--model_init_seed', type=int, default=12345)
# Embedding
#parser.add_argument('--max_position_embeddings', type=int, default=512, help='Max length of sequence')
# parser.add_argument('--embedding_dropout', type=float, default=0, help='Dropout probability during embedding')
parser.add_argument('--hidden_units', type=int, default=64, help='Size of hidden vectors (d_model)')
parser.add_argument('--num_items', type=int, default=3420, help='Number of total items')#ml-1m 3420, toys 11928, nyc16 444, chi18 250
# BERT #
parser.add_argument('--bert_intermediate_size', type=int, default=256, help='Size of intermediate feed forward')
parser.add_argument('--bert_num_layers', type=int, default=2, help='Number of transformer layers')
parser.add_argument('--bert_num_heads', type=int, default=2, help='Number of heads for multi-attention')
parser.add_argument('--bert_dropout', type=float, default=0.2, help='Dropout probability to use throughout the model')
# Mutual Learning
parser.add_argument('--base_models_name', type=str, default='bert,bert', help='Name of base models (sep by ,)')

################
# Experiment
################
parser.add_argument('--experiment_dir', type=str, default='experiments')
parser.add_argument('--experiment_description', type=str, default='ml-1m_test')

# deep evidence network
parser.add_argument('--trainer_code', type=str, default='Trainer', choices=['TrainerEDL', 'Trainer', 'FinetuningEDL'])
parser.add_argument('--dataloader_code', type=str, default='Dataloader', choices=['DataloaderEDL', 'Dataloader'])

##Loss parameters, Trainer
parser.add_argument('--alpha', type=float, default=1, help='target_weight')
parser.add_argument('--beta', type=float, default=0.1, help='non_target_weight')
parser.add_argument('--gamma', type=float, default=2.0, help='cognition weight')

parser.add_argument('--learning_pattern', default='mutual_all', choices=['separate', 'mutual_all', 'mutual_target', 'mutual_nontarget', 'mutual_target_alpha', 'mutual_target_alpha_nontarget','mutual_target_nontarget_one','DKD']) #Trainer parameter
parser.add_argument('--non_target_sampling_strategy', default='pop_first', choices=['pop_first','random','pop_last'])
parser.add_argument('--off_mainfold_eps', type=float, default=0.01)

parser.add_argument('--loss_term', type=str, default='ml') #'ml', 'cl', 'uncertainty'
parser.add_argument('--load_pretrain', type=str, default=None) #ml-1m_Standard_2022-02-19_0

parser.add_argument('--template', type=str, default=None, choices=['standard', 'evidence', 'finetuningEDL', 'mutual', 'case', 'separate'])
################

args = parser.parse_args()

# args.template = 'mutual'
# set_template(args)

args.num_items = dataset_items[args.dataset_code]