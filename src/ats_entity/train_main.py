from __future__ import unicode_literals, print_function, division
'''
    Refactor from https://github.com/huggingface/transformers/blob/master/examples/pytorch/summarization/run_summarization_no_trainer.py
'''
import os, math
import logging
import datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import transformers
from transformers import (AutoTokenizer, AdamW, set_seed)
from accelerate import Accelerator, DistributedDataParallelKwargs
from parse_args import TrainArgsParsing
from load_vocabs import load_vocabs
from data_fac import build_training_dataset
from data_collator import DataCollatorForSeq2Seq
from file_utils import get_save_load_dir
from checkpoint import Checkpoint
from scheduler_plus import get_scheduler_plus
from save_load_util import AccelerateSaveLoad
from trainer import Trainer
from validator import Validator, postprocess_text
from options_loader import TOKENIZER_FOLDER, TRAIN_OPTION
from model_choice import (import_model, exclude_struct_features)
from common.earlystop_cond import EarlyStopRougeCondition
from validator_summary_sampler import ValidationSummarySampler
from model.modelutil import query_model_size

logger = logging.getLogger(__name__)


def main():
    args = TrainArgsParsing().parse()

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    save_load = AccelerateSaveLoad(accelerator)
    args, options, regex_file_pattern = \
        save_load.load_saved_config(TRAIN_OPTION, args, logger)

    if accelerator.is_main_process:
        assert args.model_dir is not None
        save_dir = get_save_load_dir(args, options.aux_model.model_name)
        os.makedirs(save_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    logger.info("All processes are synchronized.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name,
                                                  use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # Load structure vocabs.
    vocabs = load_vocabs(args, logger)

    model = import_model(args, options, vocabs=vocabs, tokenizer=tokenizer, logger=logger)

    if args.query_model_size:
        model_size = query_model_size(model)
        logger.info(f"Model size: {model_size:.3f}MB")

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": options.training.optimizer["weight_decay"],
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    opt_method = eval(options.training.optimizer["optimizer"])
    optimizer = opt_method(optimizer_grouped_parameters,
                            lr=options.training.optimizer["lr"])

    model.seq2seq.resize_token_embeddings(len(tokenizer))
    if model.seq2seq.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    (train_dataset, eval_dataset) = build_training_dataset(args=args,
                                                        config=options.training,
                                                        accelerator=accelerator,
                                                        logger=logger)
    include_features = args.regularize_phase != "pretraining"
    exclude_feature_types = exclude_struct_features(
                                args.regularize_phase,
                                is_training=True
                            ) if args.regularize_phase != "pretraining" else None
    label_pad_token_id = -100 if options.training.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
        include_features=include_features,
        exclude_feature_types=exclude_feature_types
    )
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator,
                                batch_size=options.training.per_device_train_batch_size)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
        include_features=False,
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator,
                                batch_size=options.training.per_device_eval_batch_size)

    options.training.n_train_iterations = min(options.training.n_train_iterations,
                                             len(train_dataloader)//accelerator.state.num_processes)
    options.training.n_eval_iterations = min(options.training.n_eval_iterations,
                                            len(eval_dataloader)//accelerator.state.num_processes)

    # Work out epoch based stop bound accordingly.
    if options.training.earlystop.on:
        options.training.earlystop.stop_bound = options.training.earlystop.n_epochs * \
                                                options.training.n_train_iterations

    if args.dataset_changed:
        max_train_steps = math.ceil((options.training.max_epochs-options.training.train_state.start_epoch)
                                    *options.training.n_train_iterations / \
                                    options.training.gradient_accumulation_steps)
    else:
        max_train_steps = math.ceil((options.training.max_epochs*options.training.n_train_iterations - \
                                    options.training.train_state.train_global_count) / \
                                    options.training.gradient_accumulation_steps)
    assert max_train_steps > 0, "max_train_steps is not allowed equal to or less than zero."

    lr_scheduler = get_scheduler_plus(
        name=options.training.optimizer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=options.training.warmup.scheduler,
        num_training_steps=max_train_steps,
        num_cycles=options.training.optimizer.num_restarts,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = \
        accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader, lr_scheduler)

    # Training
    total_batch_size = options.training.per_device_train_batch_size  * \
                        accelerator.num_processes * \
                        options.training.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {options.training.n_train_iterations}")
    logger.info(f"  Num Epochs = {options.training.max_epochs}")
    logger.info(f"  Instantaneous batch size per device = {options.training.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {options.training.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    logger.info(f"  Set learning scheduler = {options.training.optimizer.lr_scheduler_type} "
                f"  with {options.training.optimizer.num_restarts} restarts")

    # Load model state etc.
    try:
        save_load.load(args, options, model, logger)
    except Exception as ex:
        logger.warning(str(ex))
        logger.warning(f"Failed to load model checkpoint. Running from scratch.")

    # Training
    log_dir = os.path.join(get_save_load_dir(args, options.aux_model.model_name), "run")
    summary_writer = SummaryWriter(log_dir) if accelerator.is_main_process else None
    checkpoint = Checkpoint(options.training.train_state, regex_file_pattern, save_load)
    stop_cond = EarlyStopRougeCondition(stop_count=args.early_stop_count_on_rouge) \
                    if args.early_stop_count_on_rouge is not None else None
    vss = ValidationSummarySampler(
                freq=args.vss_freq,
                max_freq=args.vss_max_freq,
                n_samples=args.vss_n_samples,
                validation_folder=args.vss_folder,
                postprocess_text=postprocess_text,
            ) if args.vss_max_freq > 0 else None
    validate_processor = Validator(tokenizer, stop_cond, vss)
    train_processor = Trainer(validate_processor, checkpoint, skip_except=args.skip_except)
    train_processor(args=args,
                    options=options,
                    datasets=(train_dataloader, eval_dataloader),
                    model=model,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    accelerator=accelerator,
                    max_train_steps=max_train_steps,
                    summary_writer=summary_writer,
                    logger=logger)
    if accelerator.is_main_process:
        save_dir = os.path.join(get_save_load_dir(args, options.aux_model.model_name), TOKENIZER_FOLDER)
        os.makedirs(save_dir, exist_ok=True)
        tokenizer.save_pretrained(save_dir)
    logger.info("Training is Done")


if __name__ == "__main__":
    main()
