from __future__ import unicode_literals, print_function, division

def import_model(args, options, vocabs, tokenizer, logger):
    try:
        modeling_choice = options.training.train_state["modeling_choice"]
    except Exception as ex:
        modeling_choice = None
    modeling_choice = args.modeling_choice if modeling_choice is None else modeling_choice
    assert modeling_choice == args.modeling_choice, \
            f"The specific {args.modeling_choice} is not compatible with the saved {modeling_choice}"
    if modeling_choice == "model_mrl":
        logger.info("Model is chosen from model.model_mrl")
        from model.model_mrl import Model
    else:
        raise ValueError("Chosen model name is unknown.")
    return Model(args, options, vocabs=vocabs, tokenizer=tokenizer, logger=logger)

def exclude_struct_features(regularize_phase, is_training):
    if "pretraining" in regularize_phase:
        excluded = {"encoder": ["coref"],
                    "decoder": ["coref"]} if is_training else \
                    {"encoder": ["coref", "subword"],
                    "decoder": ["coref", "subword"]}
    else:
        excluded = {"encoder": [],
                    "decoder": ["coref"]} if is_training else \
                    {"encoder": ["coref", "subword"],
                    "decoder": ["coref", "subword"]}
    return excluded
