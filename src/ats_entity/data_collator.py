from __future__ import unicode_literals, print_function, division
'''
    Modified version of https://github.com/huggingface/transformers/blob/v4.15.0/src/transformers/data/data_collator.py#L513
'''
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import (PreTrainedTokenizerBase,
                                                  BatchEncoding,
                                                  EncodedInput)


@dataclass
class DataCollatorForSeq2Seq:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    include_features: bool = False
    exclude_feature_types: Optional[Dict] = None
    return_tensors: str = "pt"
    not_pad_features = ["coref_sent_num",
                        "coref_sent_num_mask",
                        "subword_span_map",
                        "sentence_sizes",
                        "tokenized_sent_sizes"]
    split_keys = ["token_mask",
                    "subword_edge", "subword_depth", "subword_mask", "subword_span", "subword_span_map",
                    "coref_index", "coref_repr_mask", "coref_head_index",
                    "coref_entity_mask", "coref_entity_span",
                    "coref_sent_num", "coref_sent_num_mask",
                    "coref_type", "coref_number", "coref_animacy", "coref_gender",
                    "sentence_sizes", "tokenized_sent_sizes"]

    def __call__(self, batch, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        batch = list(zip(*batch)) # [(inputs, label),(inputs, label)] => [(inputs, inputs),(label, label)]
        inputs = batch[0]
        labels = batch[1] if len(batch) >= 2 else None
        indexes = batch[2] if len(batch) >= 3 else None

        # Huggingface tokenizer only pad input_ids, attention_mask and token_type_ids.
        # So, we split the other keyed data from these and pad them separately.
        struct_inputs = None
        if self.include_features:
            # Exclude data specifically excluded.
            encoder_excludes = self.exclude_feature_types.get("encoder", None)
            excludes = self.get_feature_field_names(encoder_excludes)
            id_inputs, struct_inputs, _ = self._split_dict(inputs, excludes)
            # Split data not to be padded and tensorized.
            _, struct_inputs, not_pad_inputs = self._split_dict(struct_inputs, self.not_pad_features)
            # Convert to dataframe alike.
            not_pad_inputs = {key: [example[key] for example in not_pad_inputs] \
                                    for key in not_pad_inputs[0].keys()}
        else:
            excludes = self.get_feature_field_names()
            id_inputs, _, _ = self._split_dict(inputs, excludes)
        inputs = self.tokenizer.pad(
            id_inputs,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        if self.include_features:
            try:
                attention_mask = inputs["attention_mask"]
                struct_inputs = self.pad_struct(struct_inputs,
                                                attention_mask=attention_mask,
                                                return_tensors=return_tensors)
            except:
                return (None, None, None, indexes)

            struct_inputs = {**struct_inputs, **not_pad_inputs} # Merge
        # inputs = {**inputs, **struct_inputs} # Merge

        struct_labels = None
        if labels is not None:
            if self.include_features:
                # Exclude data specifically excluded.
                decoder_excludes = self.exclude_feature_types.get("decoder", None)
                excludes = self.get_feature_field_names(decoder_excludes)
                id_labels, struct_labels, _ = self._split_dict(labels, excludes)
                # Split data not to be padded and tensorized.
                _, struct_labels, not_pad_labels = self._split_dict(struct_labels, self.not_pad_features)
                # Convert to dataframe alike.
                not_pad_labels = {key: [example[key] for example in not_pad_labels] \
                                        for key in not_pad_labels[0].keys()}
            else:
                excludes = self.get_feature_field_names()
                id_labels, _, _ = self._split_dict(labels, excludes)
            labels = self.tokenizer.pad(
                id_labels,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=return_tensors,
            )
            if self.include_features:
                try:
                    attention_mask = labels["attention_mask"]
                    struct_labels = self.pad_struct(struct_labels,
                                                    attention_mask=attention_mask,
                                                    return_tensors=return_tensors)
                except:
                    return (None, None, None, indexes)

                struct_labels = {**struct_labels, **not_pad_labels} # Merge
            # labels = {**labels, **struct_labels} # Merge
            inputs["labels"] = labels.pop("input_ids")
            inputs["decoder_attention_mask"] = labels.pop("attention_mask")

        # prepare decoder_input_ids
        decoder_input_ids = None
        if self.model is not None:
            if hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
                decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=inputs["labels"])
                inputs["decoder_input_ids"] = decoder_input_ids
            elif hasattr(self.model.seq2seq, "prepare_decoder_input_ids_from_labels"):
                decoder_input_ids = self.model.seq2seq.prepare_decoder_input_ids_from_labels(labels=inputs["labels"])
                inputs["decoder_input_ids"] = decoder_input_ids

        # return features
        if self.include_features:
            return (inputs, struct_inputs, struct_labels, indexes)
        else:
            return (inputs, indexes)

    def _split_dict(self, data, excludes=[]):
        input_data = [{k: v for k, v in d.items() if k not in self.split_keys and k not in excludes} for d in data]
        struct_data = [{k: v for k, v in d.items() if k in self.split_keys and k not in excludes} for d in data]
        exclude_data = [{k: v for k, v in d.items() if k in excludes} for d in data]
        return input_data, struct_data, exclude_data

    def get_feature_field_names(self, feature_types=None):
        feature_types = [] if feature_types is None else feature_types
        excludes = []
        if "coref" in feature_types:
            excludes += ["coref_index",
                        "coref_repr_mask",
                        "coref_entity_mask",
                        "coref_entity_span",
                        "coref_head_index",
                        "coref_sent_num",
                        "coref_sent_num_mask",
                        "coref_type",
                        "coref_number",
                        "coref_animacy",
                        "coref_gender",
                        "sentence_sizes",
                        "tokenized_sent_sizes"]
        if "subword" in feature_types:
            excludes += ["subword_edge",
                        "subword_depth",
                        "subword_mask",
                        "subword_span",
                        "subword_span_map"]
        if True: # Dataset may have fields not used for this project.
            excludes += ["pos",
                         "inarc",
                         "depth_d",
                         "head",
                         "root_head_mask"]
        return excludes

    def pad_struct(self, features, attention_mask, return_tensors):
        '''
            features: to be pad.
            attention_mask: used to guide padding.
            Note that:
                Malformed coreference feature may have entity include very long clause (e.g. which clause)
                as part of the entity. It may result in the feature data's length exceeding
                the length of text sample.
        '''
        nb, nl = attention_mask.size()

        features = {key: [example[key] for example in features] for key in features[0].keys()}

        max_length = max([max([len(example) for example in examples]) for key, examples in features.items()])
        # max_length = max(nl, max_length)
        if nl < max_length:
            raise ValueError("Abnormal struct data is found.")

        # labels_pos = [feature["labels_pos"] for feature in features] \
        #                 if "labels_pos" in features[0].keys() else None
        batch_outputs = {}
        for i in range(nb):
            inputs = dict((k, v[i]) for k, v in features.items())
            outputs = self._pad(
                inputs,
                max_length=nl,
                padding_side=self.tokenizer.padding_side,
                pad_token_id=self.tokenizer.pad_token_id,
                return_attention_mask=True
            )

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        return BatchEncoding(batch_outputs, tensor_type=return_tensors)

    def _pad(
        self,
        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
        max_length: Optional[int] = None,
        padding_side: Optional[str] = "right",
        pad_token_id: Optional[int] = None,
        return_attention_mask: Optional[bool] = None
    ) -> dict:
        masks = {}
        for k, required_input in encoded_inputs.items():
            difference = max_length - len(required_input)
            if padding_side == "right":
                if return_attention_mask:
                    masks[k+"_mask"] = [1] * len(required_input) + [0] * difference
                encoded_inputs[k] = required_input + [pad_token_id] * difference
            elif padding_side == "left":
                if return_attention_mask:
                    masks[k+"_mask"] = [0] * difference + [1] * len(required_input)
                encoded_inputs[k] = [pad_token_id] * difference + required_input
            else:
                raise ValueError("Invalid padding strategy:" + str(padding_side))
        encoded_inputs.update(masks)
        return encoded_inputs
