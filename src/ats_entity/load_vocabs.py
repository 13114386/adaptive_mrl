from __future__ import unicode_literals, print_function, division
import os, json
from collections import Counter
import torch

def load_vocab_pth(vocab_path, logger):
    fpart, _ = os.path.splitext(vocab_path)
    pth_path = f'{fpart}.pth'
    if os.path.isfile(pth_path):
        try:
            vocab = torch.load(pth_path)
            logger.info(f"Loaded vocab from {pth_path}")
            return vocab
        except:
            logger.info(f"Failed to load the vocab from {pth_path}")
            return None

def load_vocab(vocab_path, logger):
    from torchtext.vocab import vocab
    # Try to load from the pickle file.
    the_vocab = load_vocab_pth(vocab_path, logger)
    if the_vocab is None:
        # Try to load from json file.
        with open(vocab_path, "r", encoding="utf-8") as fp:
            the_vocab = json.load(fp)
        logger.info(f"Loaded vocab path: {vocab_path}")
    feat_vocab = vocab(Counter(the_vocab["freqs"]), specials=the_vocab["specials"])
    return feat_vocab

def load_vocabs(args, logger):
    vocab_names = ["coref_animacy_vocab",
                   "coref_number_vocab",
                   "coref_type_vocab",
                   "coref_gender_vocab"]
    vocab_files = [args.coref_animacy_vocab_file,
                   args.coref_number_vocab_file,
                   args.coref_type_vocab_file,
                   args.coref_gender_vocab_file]
    vocabs = {}
    for vocab_name, vocab_file in zip(vocab_names, vocab_files):
        if vocab_file is None:
            logger.info(f"{vocab_name} is not specified.")
            continue
        vocab_path = os.path.join(args.dataset_root,
                                    args.dataset_folder,
                                    vocab_file)
        vocab = load_vocab(vocab_path, logger)
        _, vocab_file = os.path.split(vocab_file)
        vocabs[vocab_file] = vocab
    return vocabs
