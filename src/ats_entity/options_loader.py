from __future__ import unicode_literals, print_function, division
import os, glob
from collections import OrderedDict
from transformers import AutoConfig as TAutoConfig
from common.auto_config import AutoConfig
from file_utils import get_save_load_dir

TOKENIZER_FOLDER="tokenizer"

TRAIN_OPTION="train"
TEST_OPTION="test"

OPTIONS = {
    TRAIN_OPTION: {
        'training': 'settings/{config_folder}/training.json',
        'base_model': 'settings/{config_folder}/bart_config.json',
        'aux_model': 'settings/{config_folder}/aux_modeling.json',
        'saveload': 'settings/{config_folder}/save_load.json',
    },
    TEST_OPTION: {
        'training': 'settings/{config_folder}/training.json',
        'base_model': 'settings/{config_folder}/bart_config.json',
        'aux_model': 'settings/{config_folder}/aux_modeling.json',
        'saveload': 'settings/{config_folder}/save_load.json',
    }
}


def parse_latest_path(fpath_pat, fpath, regex_num_pattern):
    # Locate the best file by pattern
    flist = glob.glob(fpath_pat)
    if len(flist) == 0:
        # Fall back to default path
        flist = glob.glob(fpath)
    if len(flist) == 0:
        # None can do by now
        fpath = ""
    elif len(flist) == 1:
        fpath = flist[0]
    else:
        # For multiple matches, find the latest (the highest numbering).
        import regex
        pat = regex.compile(regex_num_pattern)
        epoch_nums = [int(pat.findall(f)[0]) for f in flist]
        last_epoch = max(epoch_nums)
        max_index = epoch_nums.index(last_epoch)
        fpath = flist[max_index]
    return fpath


def parse_reload_folder(folder, file_pattern, regex_num_pattern, logger):
    fpath_pat = os.path.join(folder, file_pattern)
    fpath = parse_latest_path(fpath_pat, "", regex_num_pattern)
    logger.info(f"Reload folder: {fpath}")
    return fpath


def parse_reload_options(folder, fpath, file_pattern, regex_num_pattern, logger):
    fpath = os.path.join(folder, fpath)
    path_part, ext_part = os.path.splitext(fpath)
    fpath_pat = path_part+file_pattern+ext_part
    fpath = parse_latest_path(fpath_pat, fpath, regex_num_pattern)
    logger.info(f"Reload options: {fpath}")
    return fpath


class OptionsLoader():
    @staticmethod
    def load(mode, args, logger, verbose=True):
        options = OptionsLoader.default_options(mode, args, logger=logger)
        options = OptionsLoader.args_override(args, options, logger=logger)
        args, options, regex_file_pattern = \
            OptionsLoader.pretrained_override(args, options, logger)
        if verbose:
            logger.info(options)
        logger.info('Configuration loaded.')
        return args, options, regex_file_pattern

    @staticmethod
    def default_options(mode, args, logger):
        logger.info('Loading default configuration.')
        options = OrderedDict()
        options_frame = OPTIONS[mode]
        for key, fpath in options_frame.items():
            fpath = fpath.format(config_folder=args.config_folder)
            options[key] = AutoConfig.from_json_file(fpath)
        cfg = AutoConfig(**options)
        return cfg

    @staticmethod
    def args_override(args, options, logger):
        logger.info('Overriding by command line arguments.')
        if args.model_dir:
            options.saveload.model_dir = args.model_dir
        else:
            args.model_dir = options.saveload.model_dir
        # Override by 3party model config.
        if args.base_model_pretrained_name is not None or \
            args.base_model_config_name is not None:
            config_name = args.base_model_pretrained_name \
                        if args.base_model_pretrained_name is not None \
                        else args.base_model_config_name
            config = TAutoConfig.from_pretrained(config_name)
            options.base_model = AutoConfig(**config.to_dict())
        return options

    @staticmethod
    def pretrained_override(args, options, logger):
        logger.info('Overriding by saved trained configuration.')
        glob_file_pattern = options.saveload.glob_file_pattern
        regex_file_pattern = options.saveload.regex_file_pattern
        regex_num_pattern = options.saveload.regex_num_pattern

        baseline_option = ""
        aux_option = ""
        train_state_option = ""
        folder = get_save_load_dir(args, options.aux_model.model_name)
        if options.saveload.reload:
            baseline_option = options.saveload.baseline_option
            aux_option = options.saveload.aux_option
            train_state_option = options.saveload.train_state_option
        ckpt_fdir = parse_reload_folder(folder, glob_file_pattern, regex_num_pattern, logger)
        ckpt_fdir = ckpt_fdir if os.path.isdir(ckpt_fdir) else folder
        baseline_option = parse_reload_options(ckpt_fdir,
                                                baseline_option,
                                                glob_file_pattern,
                                                regex_num_pattern,
                                                logger)
        aux_option = parse_reload_options(ckpt_fdir,
                                            aux_option,
                                            glob_file_pattern,
                                            regex_num_pattern,
                                            logger)
        train_state_option = parse_reload_options(ckpt_fdir,
                                                train_state_option,
                                                glob_file_pattern,
                                                regex_num_pattern,
                                                logger)

        logger.info(f"Override baseline config path: {baseline_option}")
        logger.info(f"Override auxiliary config path: {aux_option}")
        logger.info(f"Override training state path: {train_state_option}")

        assert (os.path.isfile(baseline_option) and \
                os.path.isfile(aux_option) and \
                os.path.isfile(train_state_option)) or \
               (not os.path.isfile(baseline_option) and \
                not os.path.isfile(aux_option) and \
                not os.path.isfile(train_state_option))

        if os.path.isfile(baseline_option):
            options.base_model = AutoConfig.from_json_file(baseline_option)

        if os.path.isfile(aux_option):
            options.aux_model = AutoConfig.from_json_file(aux_option)

        # If tokenizer saved?
        tokenizer_option = os.path.join(folder, TOKENIZER_FOLDER)
        if os.path.isdir(tokenizer_option):
            args.tokenizer_name = tokenizer_option

        if os.path.isfile(train_state_option):
            train_state = AutoConfig.from_json_file(train_state_option)
            assert train_state.baseline_option in baseline_option
            assert train_state.aux_option in aux_option
            options.training.train_state = train_state

        return args, options, regex_file_pattern
