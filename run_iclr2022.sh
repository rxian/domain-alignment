python run_token_cls.py --dataset_name_source conll2003 --model_name_or_path bert-base-multilingual-cased --tokenizer_name bert-base-multilingual-cased --output_dir conll_zeroshot_en --device cuda

python run_token_cls.py --dataset_name_source conll2003 --dataset_name_target conll2002 --dataset_config_name_target es --num_train_steps 9086 --model_name_or_path conll_zeroshot_en --tokenizer_name bert-base-multilingual-cased --domain_alignment --use_im_weights --device cuda

python run_token_cls.py --dataset_name_source conll2003 --dataset_name_target conll2002 --dataset_config_name_target es --num_train_steps 9086 --domain_alignment --model_name_or_path conll_zeroshot_en --tokenizer_name bert-base-multilingual-cased --use_im_weights --estimate_im_weights --hard_confusion_mtx --device cuda


python run_text_cls.py --dataset_name_source amazon_reviews_multi --dataset_config_name_source en --text_column_name_source review_body --label_column_name_source stars --model_name_or_path bert-base-multilingual-cased --tokenizer_name bert-base-multilingual-cased --output_dir marc_zeroshot_en --device cuda

python run_text_cls.py --dataset_name_source amazon_reviews_multi --dataset_config_name_source en --text_column_name_source review_body --label_column_name_source stars --dataset_name_target amazon_reviews_multi --dataset_config_name_target ja --text_column_name_target review_body --label_column_name_target stars --model_name_or_path marc_zeroshot_en --tokenizer_name bert-base-multilingual-cased --domain_alignment --use_im_weights --device cuda

python run_text_cls.py --dataset_name_source amazon_reviews_multi --dataset_config_name_source en --text_column_name_source review_body --label_column_name_source stars --dataset_name_target amazon_reviews_multi --dataset_config_name_target ja --text_column_name_target review_body --label_column_name_target stars --model_name_or_path marc_zeroshot_en --tokenizer_name bert-base-multilingual-cased --domain_alignment --use_im_weights --hard_confusion_mtx --max_samples_im_weights_init 20000 --device cuda
