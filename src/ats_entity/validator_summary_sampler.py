from __future__ import unicode_literals, print_function, division
import os
from datetime import datetime
from utility.utility import saveToJson

class ValidationSummarySampler():
    def __init__(
        self,
        freq,
        max_freq,
        n_samples,
        validation_folder,
        postprocess_text,
    ):
        self.freq = freq
        self.n_samples = n_samples
        self.max_freq = max_freq
        self.postprocess_text = postprocess_text
        self.validation_folder = validation_folder
        self.error_occurred = False

    def __call__(
        self,
        iepoch,
        accelerator,
        input_ids,
        tokenizer,
        decoded_preds,
        decoded_labels
    ):
        if self.max_freq*self.freq <= iepoch+1 or self.error_occurred:
            return

        try:
            if iepoch+1 % self.freq == 0:
                sources = accelerator.pad_across_processes(
                    input_ids, dim=1, pad_index=tokenizer.pad_token_id
                )
                sources = accelerator.gather(sources).cpu().numpy()
                decoded_sources = tokenizer.batch_decode(sources, skip_special_tokens=True)
                decoded_sources, _ = self.postprocess_text(decoded_sources, None)
                text_triples = [{"source": source, "reference": lable, "summary": pred} \
                                    for source, lable, pred in zip(decoded_sources, decoded_labels, decoded_preds)]
                n_samples = min(self.n_samples, len(text_triples))
                now = datetime.now()
                result_folder = "result_" + now.strftime("%Y_%m_%d_%H_%M")
                save_dir = os.path.join(self.validation_folder, result_folder)
                os.makedirs(save_dir, exist_ok=True)
                output_name = 'valid.text.result.json'
                output_path = os.path.join(save_dir, output_name)
                saveToJson(output_path, text_triples[:n_samples])
                text_triples.clear()
        except Exception as ex:
            # Don't ruin the training.
            self.error_occurred = True
