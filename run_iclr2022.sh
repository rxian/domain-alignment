# Run ICLR 2022 IWDA and IWDA (oracle) experiments.
#
# All experiments were performed on one NVIDIA TITAN RTX GPU. On machines with smaller memory, `train_batch_size_per_domain` may need to be decreased (default is 8), accompanied with increases to `grad_accumulation_steps` (default is 1).

# NER experiments on CoNLL-2002 and 2003 datasets.
#
# Source language is en. Target languages include es, nl. (de is included in CoNLL-2003 shared task but not publically available.)
# The commands below perform transfer to es on `bert-base-multilingual-cased`.

# First train a zero-shot model by fine-tuning on en data.
python run_token_cls.py --dataset_name_source conll2003 --model_name_or_path bert-base-multilingual-cased --tokenizer_name bert-base-multilingual-cased --output_dir conll_zeroshot_en --device cuda

# Perform IWDA (oracle) for transfer to es. We assume knowledge of the target domain class distribution.
python run_token_cls.py --dataset_name_source conll2003 --dataset_name_target conll2002 --dataset_config_name_target es --num_train_steps 9086 --model_name_or_path conll_zeroshot_en --tokenizer_name bert-base-multilingual-cased --domain_alignment --use_cdan_features --use_im_weights --device cuda

# Perform IWDA with importance weight estimation for transfer to es. We don't know but estimate the target domain class distribution.
python run_token_cls.py --dataset_name_source conll2003 --dataset_name_target conll2002 --dataset_config_name_target es --num_train_steps 9086 --domain_alignment --use_cdan_features --model_name_or_path conll_zeroshot_en --tokenizer_name bert-base-multilingual-cased --use_im_weights --estimate_im_weights --hard_confusion_mtx --device cuda


# Sentiment analysis experiments on Multilingual Amazon Reviews Corpus.
#
# Source language is en. Target languages include de, es, fr, ja, zh.
# The commands below perform transfer to ja on `bert-base-multilingual-cased`.

python run_text_cls.py --dataset_name_source amazon_reviews_multi --dataset_config_name_source en --text_columns_name_source review_body --label_column_name_source stars --model_name_or_path bert-base-multilingual-cased --tokenizer_name bert-base-multilingual-cased --output_dir marc_zeroshot_en --device cuda

python run_text_cls.py --dataset_name_source amazon_reviews_multi --dataset_config_name_source en --text_columns_name_source review_body --label_column_name_source stars --dataset_name_target amazon_reviews_multi --dataset_config_name_target ja --text_columns_name_target review_body --label_column_name_target stars --model_name_or_path marc_zeroshot_en --tokenizer_name bert-base-multilingual-cased --domain_alignment --use_cdan_features --use_im_weights --device cuda

python run_text_cls.py --dataset_name_source amazon_reviews_multi --dataset_config_name_source en --text_columns_name_source review_body --label_column_name_source stars --dataset_name_target amazon_reviews_multi --dataset_config_name_target ja --text_columns_name_target review_body --label_column_name_target stars --model_name_or_path marc_zeroshot_en --tokenizer_name bert-base-multilingual-cased --domain_alignment --use_cdan_features --use_im_weights --estimate_im_weights --hard_confusion_mtx --max_samples_im_weights_init 20000 --device cuda
