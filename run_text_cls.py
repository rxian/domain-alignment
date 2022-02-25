"""Finetuning a 🤗 Transformers model for sequence classification."""

import argparse
import logging
import numpy as np

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import MODEL_MAPPING, AdamW, AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, SchedulerType, default_data_collator, get_scheduler, set_seed

from datasets import load_metric
import load_dataset_text_cls
from load_dataset_text_cls import load_raw_dataset, tokenize_raw_dataset

import iwda
import models_text_cls

logger = logging.getLogger(__name__)

# You should update this to your particular problem to have better documentation of `model_type`
# MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
# MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
MODEL_TYPES = ('bert', 'roberta', 'xlm-roberta',)
transformers.BertForSequenceClassification = models_text_cls.BertForSequenceClassification
transformers.RobertaForSequenceClassification = models_text_cls.RobertaForSequenceClassification


def get_class_dist(dataset, num_classes):
    class_dist = torch.zeros(num_classes)
    l, c = np.unique([x for y in dataset['labels'] for x in y],return_counts=True)
    if -100 in l:
        l, c = l[1:], c[1:]
    class_dist[l] = torch.tensor(c/np.sum(c)).type(class_dist.dtype)
    return class_dist


def main():
    args = parse_args()

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO)
    load_dataset_text_cls.datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_info()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Get the source domain dataset.
    raw_datasets, label_list, label_to_id = load_raw_dataset(
        dataset_name=args.dataset_name_source,
        dataset_config_name=args.dataset_config_name_source,
        train_file=args.train_file_source,
        validation_file=args.validation_file_source,
        text_column_name=args.text_column_name_source,
        label_column_name=args.label_column_name_source,
    )
    num_labels = len(label_list)

    # Metrics
    metric = load_metric("accuracy")

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
    tokenizer_name_or_path = args.tokenizer_name if args.tokenizer_name else args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
    model.to(args.device)

    # Tokenize the source domain dataset
    train_dataset_source, eval_dataset_source = tokenize_raw_dataset(
        tokenizer=tokenizer,
        raw_datasets=raw_datasets,
        label_list=label_list,
        label_to_id=label_to_id,
        text_column_name=args.text_column_name_source,
        label_column_name=args.label_column_name_source,
        pad_to_max_length=args.pad_to_max_length,
        max_length=args.max_length,
    )

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer)

    train_dataloader_source = DataLoader(train_dataset_source, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_per_domain_train_batch_size)
    eval_dataloader_source = DataLoader(eval_dataset_source, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Get, tokenize, and create DataLoaders for target domain datasets
    train_dataloader_target = None
    eval_dataloader_target = None
    if any([x is not None for x in [args.dataset_name_target, args.train_file_target, args.validation_file_target]]):
        raw_datasets, _, _ = load_raw_dataset(
            dataset_name=args.dataset_name_target,
            dataset_config_name=args.dataset_config_name_target,
            train_file=args.train_file_target,
            validation_file=args.validation_file_target,
            text_column_name=args.text_column_name_target,
            label_column_name=args.label_column_name_target,
        )
        train_dataset_target, eval_dataset_target = tokenize_raw_dataset(
            tokenizer=tokenizer,
            raw_datasets=raw_datasets,
            label_list=label_list,
            label_to_id=label_to_id,
            text_column_name=args.text_column_name_target,
            label_column_name=args.label_column_name_target,
            pad_to_max_length=args.pad_to_max_length,
            max_length=args.max_length,
        )
        train_dataloader_target = DataLoader(
            train_dataset_target, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_per_domain_train_batch_size
        )
        eval_dataloader_target = DataLoader(eval_dataset_target, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
            'lr': args.lr,
        }, {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            'lr': args.lr,
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = int(np.ceil(len(train_dataloader_source) / args.gradient_accumulation_steps))
    if args.num_train_steps is None:
        args.num_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = int(np.ceil(args.num_train_steps / num_update_steps_per_epoch))

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=int(np.ceil(args.num_train_steps*args.warmup_ratio)),
        num_training_steps=args.num_train_steps,
    )

    # Load domain adversary, and create optimizer for it
    if args.domain_alignment:
        
        # In CDAN, discriminator feature is kronecker product of model feature output and softmax output
        feature_size = model.config.hidden_size * num_labels

        if not args.use_im_weights:  
            # Vanilla domain adaptation

            model_ad = iwda.W1CriticWithImWeights(feature_size, args.hidden_size_adversary)

        else:
            # Class-importance-weighted domain adaptation

            source_class_dist = get_class_dist(train_dataset_source, num_labels).type(torch.float32)

            if not args.estimate_im_weights:
            
                if args.target_class_dist is None:
                    # Target class distribution not provided; get from labeled target dataset (for evaluating IWDA-oracle)
                    target_class_dist = get_class_dist(train_dataset_target, num_labels)
                else:
                    target_class_dist = torch.tensor(args.target_class_dist)

                model_ad = iwda.W1CriticWithImWeights(feature_size, args.hidden_size_adversary, target_class_dist/source_class_dist)
            
            else:

                im_weights_init = None

                if args.alpha_im_weights_init > 0:
                    # Initialize importance weights from model output on training datasets

                    im_weights_estimator = iwda.ImWeightsEstimator(num_labels, source_class_dist, hard_confusion_mtx=args.hard_confusion_mtx, confusion_mtx_agg_mode='mean')
                    im_weights_estimator.to(args.device)

                    # Iterate over the training datasets
                    model.eval()
                    for is_target_dom, dataloader in enumerate([train_dataloader_source, train_dataloader_target]):
                        for step, batch in enumerate(dataloader):
                            with torch.no_grad():
                                batch = {k: v.to(args.device) for k, v in batch.items()}
                                outputs = model(**batch)

                                y_true = None if is_target_dom else batch['labels']
                                y_proba = torch.nn.functional.softmax(outputs.logits,dim=-1)

                                # Collect statistics for importance weights estimation
                                im_weights_estimator(y_true=y_true, y_proba=y_proba, is_target_dom=is_target_dom)

                                # Limit on training samples for initializing importance weights
                                if args.max_samples_im_weights_init is not None and step+1 >= args.max_samples_im_weights_init:
                                    break

                    im_weights_init = im_weights_estimator.update_im_weights_qp() * args.alpha_im_weights_init + (1-args.alpha_im_weights_init)

                model_ad = iwda.W1CriticWithImWeightsEstimation(feature_size, args.hidden_size_adversary, num_labels, source_class_dist, im_weights_init=im_weights_init, hard_confusion_mtx=args.hard_confusion_mtx)

        model_ad.to(args.device)

        optimizer_ad_grouped_parameters = [
            {
                "params": [p for n, p in model_ad.named_parameters() if not 'im_weights_estimator' in n and not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay_adversary,
                'lr': args.lr_adversary,
            }, {
                "params": [p for n, p in model_ad.named_parameters() if not 'im_weights_estimator' in n and any(nd in n for nd in no_decay)], 
                "weight_decay": 0.0,
                'lr': args.lr_adversary,
            }, {
                "params": [p for n, p in model_ad.named_parameters() if 'im_weights_estimator' in n], 
                "weight_decay": 0.0,
                'lr': args.lr_im_weights,
            }
        ]
        optimizer_ad = AdamW(optimizer_ad_grouped_parameters)

    # Train!
    total_batch_size = args.per_device_per_domain_train_batch_size * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples source = {len(train_dataset_source)}")
    if train_dataloader_target is not None:
        logger.info(f"  Num examples target = {len(train_dataset_target)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_per_domain_train_batch_size}")
    logger.info(f"  Total train batch size (w. accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.num_train_steps}")
    if args.domain_alignment and args.use_im_weights:
        target_class_dist_estimate = [float("{0:0.4f}".format(i)) for i in (source_class_dist * model_ad.get_im_weights().detach().cpu()).numpy()]
        logger.info(f"  Target class distribution estimated from importance weights (init) = {target_class_dist_estimate}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.num_train_steps))
    completed_steps = 0

    train_dataloaders = (train_dataloader_source,) + ((train_dataloader_target,) if train_dataloader_target is not None else ())
    eval_dataloaders = (eval_dataloader_source,) + ((eval_dataloader_target,) if eval_dataloader_target is not None else ())

    for epoch in range(args.num_train_epochs):
        model.train()
        iterators = [iter(x) for x in train_dataloaders]
        
        step = 0
        while True:

            # Get the next batch of data from source and target domains
            batches = []
            try:
                batches.append(next(iterators[0]))
            except StopIteration:
                break            
            if len(iterators) > 1:
                try:
                    batches.append(next(iterators[1]))
                except StopIteration:
                    iterators[1] = iter(train_dataloaders[1])
                    batches.append(next(iterators[1]))

            # For keeping source and target domain discriminator features used to compute gradient penalty
            features_detached = []

            for is_target_dom, batch in enumerate(batches):
                batch = {k: v.to(args.device) for k, v in batch.items()}
                outputs = model(**batch)

                joint_loss = 0
                if not is_target_dom:
                    loss = outputs.loss
                    loss = loss / args.gradient_accumulation_steps
                    joint_loss += loss

                if args.domain_alignment:

                    y_true = None if is_target_dom else batch['labels']
                    y_proba = torch.nn.functional.softmax(outputs.logits, dim=-1).detach()
                    feature = outputs.features

                    # Get CDAN features from kronecker product of model features and output softmax
                    feature = torch.bmm(y_proba.unsqueeze(2), feature.unsqueeze(1)).view(-1,y_proba.size(1) * feature.size(1))

                    features_detached.append(feature.detach())

                    # Gradually ramp up strength of domain alignment
                    lambda_domain_alignment = args.lambda_domain_alignment
                    if args.warmup_ratio_domain_alignment is not None:
                        lambda_domain_alignment *= min(1,completed_steps/(args.num_train_steps*args.warmup_ratio_domain_alignment))
                    grl = iwda.GradientReversalLayer(lambda_domain_alignment)

                    if args.use_im_weights:
                        # `alpha` is an importance weights regularizer enabled at beginning of training
                        alpha_im_weights = 1
                        if args.warmup_ratio_im_weights is not None:
                            alpha_im_weights *= min(1,completed_steps/(args.num_train_steps*args.warmup_ratio_im_weights))
                        loss_ad = model_ad(grl(feature), y_true=y_true, is_target_dom=is_target_dom,alpha=alpha_im_weights)
                    else:
                        loss_ad = model_ad(grl(feature), y_true=y_true, is_target_dom=is_target_dom)

                    loss_ad = lambda_domain_alignment * loss_ad / args.gradient_accumulation_steps
                    joint_loss += loss_ad

                    # Update importance weights statistics and get its loss
                    if args.use_im_weights and args.estimate_im_weights:
                        model_ad.im_weights_estimator(y_true=y_true, y_proba=y_proba, is_target_dom=is_target_dom, s=args.lr_confusion_mtx)
                        loss_iw = model_ad.im_weights_estimator.get_im_weights_loss()
                        loss_iw = loss_iw / args.gradient_accumulation_steps
                        joint_loss += loss_iw
                
                joint_loss.backward()

            # Compute gradient penalty
            if args.domain_alignment:
                grad_penalty = iwda.calc_gradient_penalty(model_ad, *features_detached)
                grad_penalty = args.lambda_grad_penalty * grad_penalty / args.gradient_accumulation_steps
                grad_penalty.backward()

            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader_source) - 1:
                optimizer.step()
                if args.domain_alignment:
                    optimizer_ad.step()
                    optimizer_ad.zero_grad()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1
            
            step += 1
            if completed_steps >= args.num_train_steps:
                break

        model.eval()
        eval_metric = {}
        
        for is_target_dom, dataloader in enumerate(eval_dataloaders):

            for step, batch in enumerate(dataloader):
                batch = {k: v.to(args.device) for k, v in batch.items()}
                with torch.no_grad():
                    outputs = model(**batch)
                predictions = outputs.logits.argmax(dim=-1)
                metric.add_batch(predictions=predictions,references=batch["labels"])
            
            this_metric = metric.compute()
            eval_metric.update({k+('.target' if is_target_dom else '.source'):v for k,v in this_metric.items()})

        if args.domain_alignment and args.use_im_weights and args.estimate_im_weights:
            target_class_dist_estimate = [float("{0:0.4f}".format(i)) for i in (source_class_dist * model_ad.get_im_weights().detach().cpu()).numpy()]
            eval_metric['target_class_dist_estimate'] = target_class_dist_estimate

        print(f"epoch {epoch}:", eval_metric)

    if args.output_dir is not None:
        model.save_pretrained(args.output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")

    parser.add_argument("--dataset_name_source", type=str, default=None, help="The name of the dataset to use (via the datasets library). Source domain.")
    parser.add_argument("--dataset_config_name_source", type=str, default=None, help="The configuration name of the dataset to use (via the datasets library). Source domain.")
    parser.add_argument("--train_file_source", type=str, default=None, help="A csv or a json file containing the training data. Source domain.")
    parser.add_argument("--validation_file_source", type=str, default=None, help="A csv or a json file containing the validation data. Source domain.")
    parser.add_argument("--text_column_name_source", type=str, nargs='+', default=None, help="The column name of text to input in the file (a csv or JSON file). Source domain.")
    parser.add_argument("--label_column_name_source", type=str, default=None, help="The column name of label to input in the file (a csv or JSON file). Source domain.")

    parser.add_argument("--dataset_name_target", type=str, default=None, help="The name of the dataset to use (via the datasets library). Target domain.")
    parser.add_argument("--dataset_config_name_target", type=str, default=None, help="The configuration name of the dataset to use (via the datasets library). Target domain.")
    parser.add_argument("--train_file_target", type=str, default=None, help="A csv or a json file containing the training data. Target domain.")
    parser.add_argument("--validation_file_target", type=str, default=None, help="A csv or a json file containing the validation data. Target domain.")
    parser.add_argument("--text_column_name_target", type=str, nargs='+', default=None, help="The column name of text to input in the file (a csv or JSON file). Target domain.")
    parser.add_argument("--label_column_name_target", type=str, default=None, help="The column name of label to input in the file (a csv or JSON file). Target domain.")

    parser.add_argument("--max_length", type=int, default=512, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded if `--pad_to_max_length` is passed.")
    parser.add_argument("--pad_to_max_length", action="store_true", help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.")

    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")

    parser.add_argument("--model_name_or_path", type=str, help="Path to pretrained model or model identifier from huggingface.co/models.", required=True)
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")

    parser.add_argument("--num_train_epochs", type=int, default=4, help="Total number of training epochs to perform.")
    parser.add_argument("--num_train_steps", type=int, default=None, help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--per_device_per_domain_train_batch_size", type=int, default=8, help="Batch size (per device and domain) for the training dataloader.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size (per device) for the evaluation dataloader.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument("--lr", type=float, default=1e-5, help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay to use.")
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="linear", help="The scheduler type to use.", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Ratio of training steps for the warmup in the lr scheduler.")

    parser.add_argument("--warmup_ratio_domain_alignment", type=float, default=0.1)
    parser.add_argument("--warmup_ratio_im_weights", type=float, default=0.1)
    parser.add_argument("--tokenizer_name", type=str, default=None, help="Pretrained tokenizer name or path if not the same as model_name")

    parser.add_argument("--domain_alignment", action="store_true", help="Perform adversarial domain alignment.")
    parser.add_argument("--lambda_domain_alignment", type=float, default=0.005, help="Strength of the domain alignment.")
    parser.add_argument("--lr_adversary", type=float, default=5e-4, help="Learning rate of adversary.")
    parser.add_argument("--weight_decay_adversary", type=float, default=0.01, help="Weight decay for adversary to use.")
    parser.add_argument("--lambda_grad_penalty", type=float, default=10, help="Strength of the gradient penalty.")
    parser.add_argument("--hidden_size_adversary", type=int, default=2048, help="Width of adversarial network hidden layer.")

    parser.add_argument("--use_im_weights", action="store_true", help="Use class-importance weighting for domain adversary. If not `estimate_im_weights` and `target_class_dist` is not provided then they are inferred from labeled target domain data.")
    parser.add_argument("--target_class_dist", type=float, nargs="+", default=None, help="Target domain (training data) class prior distribution.")
    parser.add_argument("--estimate_im_weights", action="store_true", help="Estimate class-importance weights.")
    parser.add_argument("--lr_im_weights", type=float, default=5e-4, help="Learning rate for importance weights.")
    parser.add_argument("--weight_decay_im_weights", type=float, default=2, help="Strength of importance weights ell_2 regularization.")
    parser.add_argument("--lr_confusion_mtx", type=float, default=5e-3, help="Learning rate for statistics used to estimate importance weights.")
    parser.add_argument("--hard_confusion_mtx", action="store_true", help="Use hard label statistics for estimating importance weights.")
    parser.add_argument("--alpha_im_weights_init", type=float, default=0.75, help="If non-zero, replace uniformly initialized importance weights with statistics from pre-trained model by this ratio.")
    parser.add_argument("--max_samples_im_weights_init", type=int, default=None)

    parser.add_argument("--device", type=str, default="cpu")


    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None and args.train_file_source is None and args.validation_file_source is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file_source is not None:
            extension = args.train_file_source.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file_source is not None:
            extension = args.validation_file_source.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    return args


if __name__ == "__main__":
    main()
