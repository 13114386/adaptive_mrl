from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
from transformers import AutoConfig, CONFIG_MAPPING, BartConfig
from model.datautil import (
    get_keys_by_prefix
)
from model.subword_net import SubwordNet
from model.coref_mrl import CorefNet
from model.modeling_bart_mrl import BartForConditionalGeneration

sanity_check=False  # Turn on for debugging sanity check


class DataCapsule():
    def __init__(self, name, logger):
        self.name = name
        self.logger = logger
        self.iepoch = -1
        self.ready = False

    def _epoch(self, iepoch):
        if self.iepoch != iepoch:
            self.iepoch = iepoch
            self.ready = True

    epoch = property(fset=_epoch)

    def __call__(self, **kwargs):
        if self.ready:
            min_w = kwargs.pop("min_w")
            max_w = kwargs.pop("max_w")
            self.logger.info(f"({self.name} MRR) dynamic margin range: [{min_w}, {max_w}] @epoch {self.iepoch}")
            self.ready = False

class DataMonitor():
    def __init__(self, logger):
        self.logger = logger
        self.capsules = {}

    def register(self, owner: str):
        self.capsules[owner] = DataCapsule(owner, self.logger)

    def update_epoch(self, iepoch):
        for v in self.capsules.values():
            v.epoch = iepoch

    def update_data(self, owner: str, **kwargs):
        self.capsules[owner](**kwargs)


class Model(nn.Module):
    def __init__(self, args, options, vocabs, tokenizer, logger):
        super().__init__()

        self.data_monitor = DataMonitor(logger=logger)

        # Instantiate baseline module
        if args.base_model_pretrained_name is not None:
            logger.info("Creating seq2seq model from pretrained weights.")
            self.seq2seq = BartForConditionalGeneration.from_pretrained(args.base_model_pretrained_name)
        elif args.base_model_config_name is not None:
            config = AutoConfig.from_pretrained(args.base_model_config_name)
            logger.info("Creating seq2seq model from scratch using pretrained configuration.")
            self.seq2seq = BartForConditionalGeneration(config)
        elif options.base_model is not None:
            logger.info("Creating seq2seq model from configuration.")
            config = options.base_model.to_dict()
            self.seq2seq = BartForConditionalGeneration(BartConfig(**config))
        else:
            config = CONFIG_MAPPING[args.model_type]()
            logger.warning("You are instantiating a new config instance from scratch.")
            logger.info("Creating seq2seq model from scratch.")
            self.seq2seq = BartForConditionalGeneration(config)

        # Auxiliary module config
        self.subword_net = subword_net = None
        if "subword" not in options.aux_model.exclude_modules:
            subword_net = SubwordNet(options.aux_model.subword)
            # Encoder's
            if options.aux_model.struct.layering.subword.encoder in ["O"]:
                self.subword_net = subword_net
            elif options.aux_model.struct.layering.subword.encoder in ["L", "H"]:
                encoder = self.seq2seq.get_encoder()
                encoder.set_subword_net(
                    subword_net,
                    options.aux_model.struct.layering.subword.encoder
                )
            else:
                raise ValueError("Unknown subword layering configuration")
            # Decoder's
            if options.aux_model.struct.layering.subword.decoder in ["O"]:
                self.subword_net = subword_net
            elif options.aux_model.struct.layering.subword.decoder in ["L", "H"]:
                decoder = self.seq2seq.get_decoder()
                decoder.set_subword_net(
                    subword_net,
                    options.aux_model.struct.layering.subword.decoder
                )
            else:
                raise ValueError("Unknown subword layering configuration")
            logger.info("subword net instantiated")

        self.coref_mrl = coref_mrl = None
        if "coref_mrl" not in options.aux_model.exclude_modules:
            kwargs = {"cls_id": tokenizer.cls_token_id,
                      "pad_id": tokenizer.pad_token_id}
            embeddings = self.seq2seq.get_input_embeddings()
            coref_mrl = CorefNet(options.aux_model.entity_mrl,
                                attr_map=options.aux_model.struct["attr_map"],
                                type_vocab=vocabs["coref.type.vocab.json"],
                                animacy_vocab=vocabs["coref.animacy.vocab.json"],
                                number_vocab=vocabs["coref.number.vocab.json"],
                                gender_vocab=vocabs["coref.gender.vocab.json"],
                                embeddings=embeddings,
                                logger=logger,
                                data_monitor=self.data_monitor,
                                **kwargs)
            if options.aux_model.struct.layering.coref.encoder in ["O"]:
                self.coref_mrl = coref_mrl
            elif options.aux_model.struct.layering.coref.encoder in ["L", "H"]:
                encoder = self.seq2seq.get_encoder()
                encoder.set_coref_regularizer(
                    coref_mrl,
                    options.aux_model.struct.layering.coref.encoder
                )
            else:
                raise ValueError("Unknown coref regularizer layering configuration")
            logger.info("coref mrl instantiated")

    def forward(self, batch, options, iepoch):
        self.data_monitor.update_epoch(iepoch)

        inputs = batch[0]
        seq2seq_kwargs = {}
        if len(batch) == 3:
            struct_inputs = batch[1]
            struct_labels = batch[2]

            # Encoder's
            encoder_subword_inputs = get_keys_by_prefix(struct_inputs, prefix="subword", pop=False)
            encoder_coref_inputs = get_keys_by_prefix(struct_inputs, prefix="coref", pop=False)
            # Decoder's
            decoder_subword_inputs = get_keys_by_prefix(struct_labels, prefix="subword", pop=False)
            # decoder_coref_inputs = get_keys_by_prefix(struct_labels, prefix="coref", pop=False)

            encoder_sent_sizes_kwargs = {"encoder_tokenized_sent_sizes": struct_inputs["tokenized_sent_sizes"]} \
                                            if encoder_coref_inputs is not None and len(encoder_coref_inputs) > 0 \
                                            else {}
            encoder_kwargs = {
                                **{
                                    "encoder_token_mask": struct_inputs["token_mask"],
                                    "encoder_token_mask_mask": struct_inputs["token_mask_mask"],
                                    "encoder_subword_inputs": encoder_subword_inputs,
                                    "encoder_coref_inputs": encoder_coref_inputs,
                                },
                                **encoder_sent_sizes_kwargs
                            }
            seq2seq_kwargs = {
                                # Encoder's
                                **encoder_kwargs,
                                # Decoder's
                                **{
                                    "decoder_token_mask": struct_labels["token_mask"],
                                    "decoder_token_mask_mask": struct_labels["token_mask_mask"],
                                    "decoder_subword_inputs": decoder_subword_inputs,
                                    # "decoder_coref_inputs": decoder_coref_inputs,
                                }
                            }

        outputs = self.seq2seq(**inputs,
                            output_attentions=True,
                            output_hidden_states=True,
                            **seq2seq_kwargs)

        m_output = {}
        m_output["cost"] = outputs.loss.cpu()

        if self.subword_net is not None:
            h_x = self.subword_net({"data": outputs.encoder_hidden_states[-1],
                                    "attention_mask": inputs["attention_mask"],
                                    "token_head": struct_inputs["subword_edge"],
                                    "token_depth": struct_inputs["subword_depth"],
                                    "token_sparse_mask": struct_inputs["subword_mask"],
                                    "token_dense_mask": struct_inputs["subword_mask_mask"]})

            h_y = self.subword_net({"data": outputs.decoder_hidden_states[-1],
                                    "attention_mask": inputs["decoder_attention_mask"],
                                    "token_head": struct_labels["subword_edge"],
                                    "token_depth": struct_labels["subword_depth"],
                                    "token_sparse_mask": struct_labels["subword_mask"],
                                    "token_dense_mask": struct_labels["subword_mask_mask"]})
        else:
            h_x = outputs.encoder_hidden_states[-1]
            h_y = outputs.decoder_hidden_states[-1]

        if self.coref_mrl is not None \
            and options.training.warmup.coref_mrl <= iepoch <= options.training.cooldown.coref_mrl:
            (coref_cost, _) = self.coref_mrl((h_x, struct_inputs))
            m_output["cost"] += coref_cost.cpu()

        return m_output

    @torch.no_grad()
    def generate(
        self,
        batch,
        options,
        **model_kwargs,
    ):
        inputs = batch[0]

        seq2seq_kwargs = model_kwargs

        return self.seq2seq.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    **seq2seq_kwargs
                )
