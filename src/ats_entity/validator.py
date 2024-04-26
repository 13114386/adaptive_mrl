from __future__ import unicode_literals, print_function, division
'''
    Refactor from https://github.com/huggingface/transformers/blob/master/examples/pytorch/summarization/run_summarization_no_trainer.py
'''
import numpy as np
import nltk
from datasets import load_metric
import torch
from common.ml_except import EarlyStopException


def postprocess_text(preds, labels):
    if preds is not None:
        preds = [pred.strip() for pred in preds]
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    if labels is not None:
        labels = [label.strip() for label in labels]
        # rougeLSum expects newline after each sentence
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    return preds, labels

class Validator():
    def __init__(self, tokenizer, stop_cond=None, vss=None):
        self.tokenizer = tokenizer
        self.stop_cond = stop_cond
        self.vss = vss
        # Metric
        self.metric = load_metric("rouge")

    def __call__(self,
            iepoch,
            ibatch,
            options,
            val_dataloader,
            model,
            accelerator,
            glb_count,
            summary_writer,
            logger):
        '''
            Evaluate by ROUGE metrics
        '''
        model.eval()

        gen_kwargs = {
            "max_length": options.training.max_target_length,
            "min_length": options.training.min_target_length,
            "num_beams": options.training.num_beams,
            "length_penalty": options.training.length_penalty
        }

        for idx, batch in zip(range(options.training.n_eval_iterations), val_dataloader):
            with torch.no_grad():
                generated_tokens = accelerator.unwrap_model(model).generate(
                    batch=batch,
                    options=options,
                    **gen_kwargs,
                )

                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=self.tokenizer.pad_token_id
                )

                inputs = batch[0]
                labels = inputs["labels"]
                if not options.training.pad_to_max_length:
                    # If we did not pad to max length, we need to pad the labels too
                    labels = accelerator.pad_across_processes(labels, dim=1,
                                                              pad_index=self.tokenizer.pad_token_id)

                generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
                labels = accelerator.gather(labels).cpu().numpy()

                if options.training.ignore_pad_token_for_loss:
                    # Replace -100 in the labels as we can't decode them.
                    labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]
                decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
                decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
                if self.vss is not None:
                    self.vss(iepoch=iepoch,
                        accelerator=accelerator,
                        input_ids=inputs["input_ids"],
                        tokenizer=self.tokenizer,
                        decoded_preds=decoded_preds,
                        decoded_labels=decoded_labels
                    )
                self.metric.add_batch(predictions=decoded_preds, references=decoded_labels)
        result = self.metric.compute(use_stemmer=True)
        # Extract a few results from ROUGE
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        result = {k: round(v, 4) for k, v in result.items()}
        fmt_result = f"Eval: Epoch {iepoch}: " + \
                    ", ".join("{}: {:.4f}".format(k, v) for k, v in result.items())
        logger.info(fmt_result)
        if summary_writer:
            summary_writer.add_scalar("ROUGE-1 F-measure",
                                      result["rouge1"],
                                      glb_count)
            summary_writer.add_scalar("ROUGE-2 F-measure",
                                      result["rouge2"],
                                      glb_count)
            summary_writer.add_scalar("ROUGE-L F-measure",
                                      result["rougeL"],
                                      glb_count)
        if self.stop_cond is not None \
           and self.stop_cond(result, logger):
            raise EarlyStopException("Early stop on ROUGE scores.")
