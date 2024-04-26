from __future__ import unicode_literals, print_function, division
import os, json
import torch
from torch.utils.data import Dataset

REMOVE_KEYS = ["pos", "inarc", "depth_d", "head", "root_head_mask", "doc"]

class CNNDMDataset(Dataset):
    def __init__(self, args, split_type, config, logger, accelerator):
        super().__init__()
        atcl_name, hlit_name = args.pair_type
        input_path = os.path.join(args.dataset_root,
                                  args.dataset_folder,
                                  args.dataset_file.format(split_type=split_type, pair_type=atcl_name))
        label_path = os.path.join(args.dataset_root,
                                  args.dataset_folder,
                                  args.dataset_file.format(split_type=split_type, pair_type=hlit_name))
        logger.info(f"Loading input data: {input_path}")
        self.input_data = self._load_from(input_path, args.dataset_build_pickle, logger)
        logger.info(f"Loading label data: {label_path}")
        self.label_data = self._load_from(label_path, args.dataset_build_pickle, logger)
        logger.info(f"Datasets are loaded.")
        assert len(self.input_data) == len(self.label_data)

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        inputs = self.input_data[idx]
        labels = self.label_data[idx]
        return (inputs, labels, idx)

    def _load_from(self, input_path, build_pickle, logger):
        # Load from pickled file if no need to rebuild pickled version.
        if not build_pickle:
            data = self._load_from_pth(input_path, logger)
            if data is not None:
                return data
        # Fall through to load from text file and build a pickled file.
        data = []
        with open(input_path, "r", encoding="utf-8") as fp:
            for index, line in enumerate(fp):
                rec = json.loads(line)
                for key in REMOVE_KEYS:
                    if key in rec:
                        del rec[key] # Reduce memory footprint.
                data.append(rec)
        self._save_to_pth(input_path, data, logger)
        return data

    def _load_from_pth(self, input_path, logger):
        fpart, fext = os.path.splitext(input_path)
        pth_path = f"{fpart}.pth"
        logger.info(f"Loading pickled {pth_path}")
        data = None
        if os.path.isfile(pth_path):
            data = torch.load(pth_path)
            logger.info(f"Loaded pickled {pth_path}")
        return data

    def _save_to_pth(self, input_path, data, logger):
        fpart, fext = os.path.splitext(input_path)
        pth_path = f"{fpart}.pth"
        torch.save(data, pth_path)
