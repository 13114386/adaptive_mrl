from __future__ import unicode_literals, print_function, division
'''
    Refactor from https://github.com/huggingface/transformers/blob/master/examples/pytorch/summarization/run_summarization_no_trainer.py
'''
import os
from datetime import datetime
import logging
import datasets
from torch.utils.data import DataLoader
import transformers
from transformers import (AutoTokenizer,
                          MODEL_MAPPING)
from accelerate import Accelerator, DistributedDataParallelKwargs
from parse_args import TestArgsParsing
from options_loader import TEST_OPTION
from load_vocabs import load_vocabs
from data_fac import build_test_dataset
from utility.utility import saveToJson
from data_collator import DataCollatorForSeq2Seq # Use default since no extra work for inference run.
from save_load_util import AccelerateSaveLoad
from tester import Tester
from tool.dupcount import ReplicationStatistic
from model_choice import (import_model)

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

logger = logging.getLogger(__name__)


def main():
    args = TestArgsParsing().parse()

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

    save_load = AccelerateSaveLoad(accelerator)
    args, options, regex_file_pattern = \
        save_load.load_saved_config(TEST_OPTION, args, logger)

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

    vocabs = load_vocabs(args, logger)

    model = import_model(args, options, vocabs=vocabs, tokenizer=tokenizer, logger=logger)

    model.seq2seq.resize_token_embeddings(len(tokenizer))
    if model.seq2seq.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    test_dataset = build_test_dataset(args=args,
                                    config=options.training,
                                    accelerator=accelerator,
                                    logger=logger)

    label_pad_token_id = -100 if options.training.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
        include_features=False,
    )

    test_dataloader = DataLoader(test_dataset,
                                shuffle=False,
                                collate_fn=data_collator,
                                batch_size=args.test_batch_size)

    # Prepare everything with our `accelerator`.
    model, test_dataloader = accelerator.prepare(
        model, test_dataloader
    )

    # Load model state etc.
    save_load.load(args, options, model, logger)

    # Test
    processor = Tester(tokenizer)
    result, text_triples = processor(options=options,
                                    dataloader=test_dataloader,
                                    model=model,
                                    accelerator=accelerator,
                                    logger=logger)
    if accelerator.is_main_process:
        # Format F score results
        f_scores = {}
        f_scores["format_Fscore_low"] = {key: round(value.low.fmeasure*100, 4) \
                                            for key, value in result.items()}
        f_scores["format_Fscore_mid"] = {key: round(value.mid.fmeasure*100, 4) \
                                            for key, value in result.items()}
        f_scores["format_Fscore_high"] = {key: round(value.high.fmeasure*100, 4) \
                                            for key, value in result.items()}

        # Save
        now = datetime.now()
        result_folder = "result_" + now.strftime("%Y_%m_%d_%H_%M")
        save_dir = os.path.join(args.evaluation_folder, result_folder)
        os.makedirs(save_dir, exist_ok=True)
        # Save F scores
        fpath = os.path.join(save_dir, 'rouge_fscores.json')
        saveToJson(fpath, f_scores)
        # Save rouge results
        fpath = os.path.join(save_dir, 'rouge_results.json')
        with open(fpath, 'w') as fp:
            print(result, file=fp)
            print(result)

        output_name = 'eval.text.result.json'
        output_path = os.path.join(save_dir, output_name)
        saveToJson(output_path, text_triples)
        # Token replication statistic
        ReplicationStatistic.compute(save_dir, text_triples, 'eval.summary')

    logger.info('Finish Testing')


if __name__ == '__main__':
    main()
