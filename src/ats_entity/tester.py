from __future__ import unicode_literals, print_function, division
'''
    Refactor from https://github.com/huggingface/transformers/blob/master/examples/pytorch/summarization/run_summarization_no_trainer.py
'''
import numpy as np
from tqdm.auto import tqdm
import nltk
from datasets import load_metric
import torch

def postprocess_text_strip(inputs):
    inputs = [x.strip() for x in inputs]
    return inputs

def postprocess_text_newline(inputs):
    inputs = ["\n".join(nltk.sent_tokenize(x)) for x in inputs]
    return inputs


class Tester():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # Metric
        self.metric = load_metric("rouge")

    def __call__(self, options, dataloader, model, accelerator, logger):
        model.eval()

        gen_kwargs = {
            "max_length": options.training.max_target_length,
            "min_length": options.training.min_target_length,
            "num_beams": options.training.num_beams,
            "length_penalty": options.training.length_penalty
        }

        max_train_steps = min(options.training.n_test_iterations, len(dataloader))
        progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)

        text_triples = []
        for idx, batch in zip(range(max_train_steps), dataloader):
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
                sources = accelerator.pad_across_processes(
                    inputs["input_ids"], dim=1, pad_index=self.tokenizer.pad_token_id
                )

                labels = inputs["labels"]
                if not options.training.pad_to_max_length:
                    # If we did not pad to max length, we need to pad the labels too
                    labels = accelerator.pad_across_processes(labels, dim=1,
                                                              pad_index=self.tokenizer.pad_token_id)

                generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
                sources = accelerator.gather(sources).cpu().numpy()
                labels = accelerator.gather(labels).cpu().numpy()

                if options.training.ignore_pad_token_for_loss:
                    # Replace -100 in the labels as we can't decode them.
                    labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]
                decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
                decoded_sources = self.tokenizer.batch_decode(sources, skip_special_tokens=True)
                # Postprocess
                decoded_sources = postprocess_text_strip(decoded_sources)
                decoded_labels = postprocess_text_strip(decoded_labels)
                decoded_preds = postprocess_text_strip(decoded_preds)
                text_triples += [{"source": source, "reference": lable, "summary": pred} \
                                    for source, lable, pred in zip(decoded_sources, decoded_labels, decoded_preds)]
                # rougeLSum expects newline after each sentence
                decoded_labels = postprocess_text_newline(decoded_labels)
                decoded_preds = postprocess_text_newline(decoded_preds)
                self.metric.add_batch(predictions=decoded_preds, references=decoded_labels)
                progress_bar.update(1)
        # Compute metric
        result = self.metric.compute(use_stemmer=True)
        return result, text_triples
