"""
Finetuning a 🤗 Transformers model for sequence classification.

Adapted from https://github.com/huggingface/transformers/blob/master/examples/pytorch/text-classification/run_glue_no_trainer.py
"""

import argparse
import logging
import numpy as np

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import MODEL_MAPPING, AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, SchedulerType, default_data_collator, get_scheduler, set_seed

from datasets import load_metric
import load_dataset_text_cls
from load_dataset_text_cls import load_raw_dataset, tokenize_raw_dataset

import domain_alignment
import models_text_cls

logger = logging.getLogger(__name__)

# Tweak models for sequence classification in 🤗 Transformers to 
# return tanh-pooled features provided to the linear classification 
# head in each forward pass so that we can perform domain alignment.
#
# The tweaked models are defined in `models_text_cls.py`, and they 
# replace the original models in Transformers library.
MODEL_TYPES = ('bert', 'roberta', 'xlm-roberta',)
transformers.models.bert.modeling_bert.BertForSequenceClassification = models_text_cls.BertForSequenceClassification
transformers.models.roberta.modeling_roberta.RobertaForSequenceClassification = models_text_cls.RobertaForSequenceClassification


def get_class_dist(dataset, num_classes):
    # Counts labels in the `dataset` and returns the class distribution.
    class_dist = torch.zeros(num_classes)
    l, c = np.unique(dataset['labels'],return_counts=True)
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

    if args.disable_tqdm:
        # https://stackoverflow.com/a/67238486/7112125
        from functools import partialmethod
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Get the source domain dataset.
    raw_datasets, label_list, label_to_id = load_raw_dataset(
        dataset_name=args.dataset_name_source,
        dataset_config_name=args.dataset_config_name_source,
        train_file=args.train_file_source,
        evaluation_file=args.evaluation_file_source,
        text_column_name=args.text_columns_name_source,
        label_column_name=args.label_column_name_source,
    )
    num_labels = len(label_list)

    # Metrics
    metric = load_metric("accuracy")

    # Load pre-trained model and tokenizer
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
        text_column_name=args.text_columns_name_source,
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
        # the samples passed).
        data_collator = DataCollatorWithPadding(tokenizer)

    train_dataloader_source = DataLoader(train_dataset_source, shuffle=True, collate_fn=data_collator, batch_size=args.train_batch_size_per_domain)
    eval_dataloader_source = DataLoader(eval_dataset_source, collate_fn=data_collator, batch_size=args.eval_batch_size)

    # Get, tokenize, and create DataLoaders for target domain datasets
    train_dataloader_target = None
    eval_dataloader_target = None
    if any([x is not None for x in [args.dataset_name_target, args.train_file_target, args.evaluation_file_target]]):
        raw_datasets, _, _ = load_raw_dataset(
            dataset_name=args.dataset_name_target,
            dataset_config_name=args.dataset_config_name_target,
            train_file=args.train_file_target,
            evaluation_file=args.evaluation_file_target,
            text_column_name=args.text_columns_name_target,
            label_column_name=args.label_column_name_target,
        )
        train_dataset_target, eval_dataset_target = tokenize_raw_dataset(
            tokenizer=tokenizer,
            raw_datasets=raw_datasets,
            label_list=label_list,
            label_to_id=label_to_id,
            text_column_name=args.text_columns_name_target,
            label_column_name=args.label_column_name_target,
            pad_to_max_length=args.pad_to_max_length,
            max_length=args.max_length,
        )
        train_dataloader_target = DataLoader(
            train_dataset_target, shuffle=True, collate_fn=data_collator, batch_size=args.train_batch_size_per_domain
        )
        eval_dataloader_target = DataLoader(eval_dataset_target, collate_fn=data_collator, batch_size=args.eval_batch_size)

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
    num_update_steps_per_epoch = int(np.ceil(len(train_dataloader_source) / args.grad_accumulation_steps))
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

    # Create domain adversary, and its optimizer
    model_ad = None
    im_weights_estimator = None
    optimizer_ad = None
    if args.domain_alignment:

        feature_size = model.config.hidden_size
        if args.use_cdan_features:
            # In CDAN, discriminator feature is kronecker product of model feature output and softmax output
            feature_size *= num_labels

        im_weights = None
        source_class_dist = get_class_dist(train_dataset_source, num_labels).type(torch.float32)

        if args.use_im_weights and not args.estimate_im_weights:
            # Class-importance-weighted domain adaptation with oracle IW
            if args.target_class_dist is not None:
                # Importance weights are provided by the user (oracle)
                target_class_dist = torch.tensor(args.target_class_dist)
            else:
                # Target class distribution not provided; get from labeled target dataset (for evaluating IWDA-oracle)
                target_class_dist = get_class_dist(train_dataset_target, num_labels)
            im_weights = target_class_dist/source_class_dist

        if args.domain_alignment_loss == 'w1':
            model_ad = domain_alignment.W1CriticWithImWeights(feature_size, args.hidden_size_adversary, im_weights=im_weights)
        elif args.domain_alignment_loss == 'jsd':
            model_ad = domain_alignment.JSDAdversaryWithImWeights(feature_size, args.hidden_size_adversary, im_weights=im_weights)
        elif args.domain_alignment_loss == 'mmd':
            model_ad = domain_alignment.MMDWithImWeights(im_weights=im_weights, kernel_mul=args.mmd_kernel_mul, kernel_num=args.mmd_kernel_num, fix_sigma=args.mmd_fix_sigma)
        model_ad.to(args.device)

        if args.use_im_weights and args.estimate_im_weights:
            # Class-importance-weighted domain adaptation with IW estimated on-the-fly

            im_weights_init = None

            if args.alpha_im_weights_init > 0:
                # Initialize importance weights from model output on training datasets
                im_weights_estimator = domain_alignment.ImWeightsEstimator(num_labels, source_class_dist, hard_confusion_mtx=args.hard_confusion_mtx, confusion_mtx_agg_mode='mean')
                im_weights_estimator.to(args.device)

                # Iterate over the training datasets, and feed model outputs to IW estimator
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

                            # Limit num of training samples used to estimate importance weights
                            if args.max_samples_im_weights_init is not None and step+1 >= args.max_samples_im_weights_init:
                                break

                # Get regularized importance weights
                im_weights_init = im_weights_estimator.update_im_weights_qp() * args.alpha_im_weights_init + (1-args.alpha_im_weights_init)

            im_weights_estimator = domain_alignment.ImWeightsEstimator(num_labels, source_class_dist, im_weights_init=im_weights_init.detach().cpu(), hard_confusion_mtx=args.hard_confusion_mtx)
            im_weights_estimator.to(args.device)
            model_ad.get_im_weights = im_weights_estimator.get_im_weights

        optimizer_ad_grouped_parameters = []
        if args.domain_alignment_loss in ['w1','jsd']:
            optimizer_ad_grouped_parameters.extend([
                {
                    "params": [p for n, p in model_ad.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": args.weight_decay_adversary,
                    'lr': args.lr_adversary,
                }, {
                    "params": [p for n, p in model_ad.named_parameters() if any(nd in n for nd in no_decay)], 
                    "weight_decay": 0.0,
                    'lr': args.lr_adversary,
                }
            ])
        if im_weights_estimator is not None:
            optimizer_ad_grouped_parameters.extend([
                {
                    "params": [p for n, p in im_weights_estimator.named_parameters()], 
                    "weight_decay": args.weight_decay_im_weights,
                    'lr': args.lr_im_weights,
                }
            ])
        if optimizer_ad_grouped_parameters:
            optimizer_ad = AdamW(optimizer_ad_grouped_parameters)

    # Train!
    total_batch_size = args.train_batch_size_per_domain * args.grad_accumulation_steps

    logger.info(f"Run arguments: {vars(args)}")
    logger.info("***** Running training *****")
    logger.info(f"  Num examples source = {len(train_dataset_source)}")
    if train_dataloader_target is not None:
        logger.info(f"  Num examples target = {len(train_dataset_target)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size_per_domain}")
    logger.info(f"  Total train batch size (w. accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.grad_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.num_train_steps}")
    if args.domain_alignment and args.use_im_weights:
        target_class_dist_estimate = [float("{0:0.4f}".format(i)) for i in (source_class_dist * model_ad.get_im_weights().detach().cpu()).numpy()]
        logger.info(f"  Estimated target class distribution (init) = {target_class_dist_estimate}")

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

            # Keep the features and labels for domain alignment
            features = []
            source_dom_labels = None

            joint_loss = 0

            for is_target_dom, batch in enumerate(batches):
                batch = {k: v.to(args.device) for k, v in batch.items()}
                outputs = model(**batch)

                if not is_target_dom:
                    loss = outputs.loss
                    loss = loss / args.grad_accumulation_steps
                    joint_loss += loss

                if args.domain_alignment:
                    # Update importance weights statistics and get its loss
                    y_proba = torch.nn.functional.softmax(outputs.logits, dim=-1).detach()
                    y_true = None if is_target_dom else batch['labels']
                    if im_weights_estimator is not None:
                        im_weights_estimator(y_true=y_true, y_proba=y_proba, is_target_dom=is_target_dom, s=args.lr_confusion_mtx)
                        loss_iw = im_weights_estimator.get_im_weights_loss()
                        loss_iw = loss_iw / args.grad_accumulation_steps
                        joint_loss += loss_iw

                    # Get features for domain alignment
                    feature = outputs.features
                    if args.use_cdan_features:
                        # CDAN features are kronecker product of model features and output softmax
                        feature = torch.bmm(y_proba.unsqueeze(2), feature.unsqueeze(1)).view(-1,y_proba.size(1) * feature.size(1))

                    features.append(feature)
                    if not is_target_dom:
                        source_dom_labels = y_true

            if args.domain_alignment:
                
                domain_labels = torch.cat([torch.zeros(len(features[0])).long(),torch.ones(len(features[1])).long()]).to(args.device)
                features_concat = torch.cat(features, dim=0)

                # Gradually ramp up strength of domain alignment
                lambda_domain_alignment = args.lambda_domain_alignment
                if args.warmup_ratio_domain_alignment > 0:
                    lambda_domain_alignment *= min(1,completed_steps/(args.num_train_steps*args.warmup_ratio_domain_alignment))

                if args.domain_alignment_loss == 'mmd':
                    lambda_domain_alignment *= -1
                features_concat = domain_alignment.GradientReversalLayer(lambda_domain_alignment)(features_concat)

                # `alpha` is an importance weights regularizer in early stages of training
                alpha_im_weights = 1
                if args.warmup_ratio_im_weights > 0:
                    alpha_im_weights *= min(1,completed_steps/(args.num_train_steps*args.warmup_ratio_im_weights))

                loss_ad = model_ad(features_concat, domain_labels, y_true=source_dom_labels, alpha=alpha_im_weights)
                loss_ad = loss_ad / args.grad_accumulation_steps
                joint_loss += loss_ad

                # Compute gradient penalty
                if args.domain_alignment_loss != 'mmd' and args.lambda_grad_penalty > 0:
                    grad_penalty = domain_alignment.calc_gradient_penalty(model_ad.net, *[feature.detach() for feature in features])
                    grad_penalty = args.lambda_grad_penalty * grad_penalty / args.grad_accumulation_steps
                    joint_loss += grad_penalty
                
            # Back-propagate the (joint) source classification (and domain alignment) loss
            joint_loss.backward()

            # Update parameters
            if step % args.grad_accumulation_steps == 0 or step == len(train_dataloader_source) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                if optimizer_ad is not None:
                    optimizer_ad.step()
                    optimizer_ad.zero_grad()

                progress_bar.update(1)
                completed_steps += 1
                if completed_steps >= args.num_train_steps:
                    break
            
            step += 1

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
            eval_metric.update({k+('/target' if is_target_dom else '/source'):v for k,v in this_metric.items()})

        # Print estimated target domain class distribution
        if args.domain_alignment and args.use_im_weights and args.estimate_im_weights:
            target_class_dist_estimate = [float("{0:0.4f}".format(i)) for i in (source_class_dist * model_ad.get_im_weights().detach().cpu()).numpy()]
            eval_metric['target_class_dist_estimate'] = target_class_dist_estimate

        logger.info(f"epoch {epoch+1}: {eval_metric}")

    if args.output_dir is not None:
        model.save_pretrained(args.output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")

    parser.add_argument("--dataset_name_source", type=str, default=None, help="The name of the dataset to use (via the datasets library). Source domain.")
    parser.add_argument("--dataset_config_name_source", type=str, default=None, help="The configuration name of the dataset to use (via the datasets library). Source domain.")
    parser.add_argument("--train_file_source", type=str, default=None, help="A csv or a json file containing the training data. Source domain.")
    parser.add_argument("--evaluation_file_source", type=str, default=None, help="A csv or a json file containing the evaluation data. Source domain.")
    parser.add_argument("--text_columns_name_source", type=str, nargs='+', default=None, help="The column names of text to input in the file (a csv or JSON file). Source domain.")
    parser.add_argument("--label_column_name_source", type=str, default=None, help="The column name of label to input in the file (a csv or JSON file). Source domain.")

    parser.add_argument("--dataset_name_target", type=str, default=None, help="The name of the dataset to use (via the datasets library). Target domain.")
    parser.add_argument("--dataset_config_name_target", type=str, default=None, help="The configuration name of the dataset to use (via the datasets library). Target domain.")
    parser.add_argument("--train_file_target", type=str, default=None, help="A csv or a json file containing the training data. Target domain.")
    parser.add_argument("--evaluation_file_target", type=str, default=None, help="A csv or a json file containing the evaluation data. Target domain.")
    parser.add_argument("--text_columns_name_target", type=str, nargs='+', default=None, help="The column names of text to input in the file (a csv or JSON file). Target domain.")
    parser.add_argument("--label_column_name_target", type=str, default=None, help="The column name of label to input in the file (a csv or JSON file). Target domain.")

    parser.add_argument("--max_length", type=int, default=512, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded if `--pad_to_max_length` is passed.")
    parser.add_argument("--pad_to_max_length", action="store_true", help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.")

    parser.add_argument("--model_name_or_path", type=str, help="Path to pre-trained model or model identifier from huggingface.co/models.", required=True)
    parser.add_argument("--tokenizer_name", type=str, default=None, help="Pre-trained tokenizer name or path if not the same as model_name")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")

    parser.add_argument("--num_train_epochs", type=int, default=4, help="Total number of training epochs to perform.")
    parser.add_argument("--num_train_steps", type=int, default=None, help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--train_batch_size_per_domain", type=int, default=8, help="Batch size (per domain) for the training dataloader.")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="Batch size for the evaluation dataloader.")
    parser.add_argument("--grad_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument("--lr", type=float, default=1e-5, help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay to use.")
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="linear", help="The scheduler type to use.", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Ratio of training steps for the warmup in the lr scheduler.")

    parser.add_argument("--domain_alignment", action="store_true", help="Perform adversarial domain alignment.")
    parser.add_argument("--domain_alignment_loss", type=str, default='w1', choices=['w1','jsd','mmd'], help="Loss for domain alignment.")
    parser.add_argument("--lambda_domain_alignment", type=float, default=5e-3, help="Strength of the domain alignment.")
    parser.add_argument("--warmup_ratio_domain_alignment", type=float, default=0.1, help="Ratio of training steps for warming up the strength of domain alignment.")
    parser.add_argument("--use_cdan_features", action="store_true", help="Use CDAN features (Long et al., 2018).")

    parser.add_argument("--lr_adversary", type=float, default=5e-4, help="Learning rate of adversary. (Only applicable if `domain_alignment_loss` is `w1` or `jsd`.)")
    parser.add_argument("--weight_decay_adversary", type=float, default=0.01, help="Weight decay for adversary to use. (Only applicable if `domain_alignment_loss` is `w1` or `jsd`.)")
    parser.add_argument("--lambda_grad_penalty", type=float, default=10, help="Strength of the gradient penalty. (Only applicable if `domain_alignment_loss` is `w1` or `jsd`.)")
    parser.add_argument("--hidden_size_adversary", type=int, default=2048, help="Width of adversarial network hidden layer. (Only applicable if `domain_alignment_loss` is `w1` or `jsd`.)")

    parser.add_argument("--mmd_kernel_num", type=int, default=5, help="Number of kernels in the MMD layer. (Only applicable if `domain_alignment_loss` is `mmd`.)")
    parser.add_argument("--mmd_kernel_mul", type=float, default=2.0, help='Multiplicative factor of kernel bandwidth. (Only applicable if `domain_alignment_loss` is `mmd`.)')
    parser.add_argument("--mmd_fix_sigma", type=float, default=None, help="Fix kernel bandwidth, otherwise dynamically adjusted according to l2 distance between pairs of data. (Only applicable if `domain_alignment_loss` is `mmd`.)")

    parser.add_argument("--use_im_weights", action="store_true", help="Use class-importance weighting for domain adversary. If not `estimate_im_weights` and `target_class_dist` is not provided then they are inferred from labeled target domain data.")
    parser.add_argument("--target_class_dist", type=float, nargs="+", default=None, help="Target domain (training data) class prior distribution.")
    parser.add_argument("--estimate_im_weights", action="store_true", help="Estimate class-importance weights.")
    parser.add_argument("--lr_im_weights", type=float, default=5e-4, help="Learning rate for importance weights.")
    parser.add_argument("--weight_decay_im_weights", type=float, default=2, help="Strength of importance weights ell_2 regularization.")
    parser.add_argument("--warmup_ratio_im_weights", type=float, default=0.1, help="Ratio of training steps for reducing the regularization on importance weights, as initial estimates could be inaccurate.")
    parser.add_argument("--lr_confusion_mtx", type=float, default=5e-3, help="Learning rate for statistics used to estimate importance weights.")
    parser.add_argument("--hard_confusion_mtx", action="store_true", help="Use hard label statistics for estimating importance weights.")
    parser.add_argument("--alpha_im_weights_init", type=float, default=0.75, help="If non-zero, replace uniformly initialized importance weights with statistics from pre-trained model by this ratio.")
    parser.add_argument("--max_samples_im_weights_init", type=int, default=None, help="Max number of training samples to use for initializing importance weight estimates.")

    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to train on, e.g. `cpu` or `cuda`.")
    parser.add_argument("--disable_tqdm", action="store_true", help="Silence `tqdm` progress bars.")

    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name_source is None and args.train_file_source is None and args.evaluation_file_source is None:
        raise ValueError("Need either a task name or a training/evaluation file.")
    else:
        if args.train_file_source is not None:
            extension = args.train_file_source.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.evaluation_file_source is not None:
            extension = args.evaluation_file_source.split(".")[-1]
            assert extension in ["csv", "json"], "`evaluation_file` should be a csv or a json file."

    return args


if __name__ == "__main__":
    main()
