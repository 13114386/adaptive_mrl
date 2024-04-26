from __future__ import unicode_literals, print_function, division
import os
from abc import ABC, abstractmethod
import torch
from accelerate.utils import RNG_STATE_NAME # accelerate version 0.6.0+
from common.auto_config import AutoConfig
from file_utils import get_save_load_dir
from options_loader import OptionsLoader


def extract_base_filename(fpath, file_pattern):
    import regex
    pat = regex.compile(file_pattern)
    m = pat.findall(fpath)
    if m:
        print(m)
        return m[-1][0]+m[-1][1]
    else:
        return fpath


class SaveLoadBase(ABC):
    @abstractmethod
    def save(
        self,
        epoch_index,
        batch_index,
        args,
        options,
        model,
        optimizers,
        best_score,
        batch_count,
        train_global_count,
        folder='epoch_batch',
        reason=None,
        regex_file_pattern=None,
        logger=None
    ):
        raise NotImplemented("SaveLoadBase not implemented.")

    def make_id(self, epoch_index, batch_index, method):
        if method == 'best_epoch_batch':
            str_id = '_best_epoch_'+str(epoch_index)+'_batch_'+str(batch_index)
        elif method == 'epoch_batch':
            str_id = '_epoch_'+str(epoch_index)+'_batch_'+str(batch_index)
        else:
            raise ValueError("Unknown checkpoint saving method.")
        return str_id

    def get_save_dir(self, args, model_name, logger, str_id=""):
        save_folder = os.path.join(get_save_load_dir(args, model_name), str_id)
        os.makedirs(save_folder, exist_ok=True)
        logger.info(f"Save directory: {save_folder}")
        return save_folder

    def save_config_state(
        self,
        epoch_index,
        batch_index,
        args,
        options,
        best_score,
        batch_count,
        train_global_count,
        save_folder,
        reason,
        str_id,
        regex_file_pattern,
        logger
    ):
        #== Save training configurations ==#
        # Baseline configuration
        _, baseline_config_name = os.path.split(options.saveload.baseline_option)
        config_path = os.path.join(save_folder, baseline_config_name)
        options.base_model.to_json_file(config_path, use_diff=False)
        logger.info(f"Model config is saved to {config_path}")

        # Auxiliary configurations
        _, aux_config_name = os.path.split(options.saveload.aux_option)
        config_path = os.path.join(save_folder, aux_config_name)
        options.aux_model.to_json_file(config_path, use_diff=False)
        logger.info(f"Aux config is saved to {config_path}")

        # Training state
        # Only need to save whatever is changed during training
        train_state = AutoConfig(**{"start_epoch": epoch_index+1,
                                    "train_global_count": train_global_count,
                                    "best_score": best_score,
                                    "batch_count": batch_count,
                                    "baseline_option": baseline_config_name,
                                    "aux_option": aux_config_name,
                                    "model_id": str_id,
                                    "modeling_choice": args.modeling_choice})
        _, train_state_file = os.path.split(options.saveload.train_state_option)
        train_state_path = os.path.join(save_folder, train_state_file)
        train_state.to_json_file(train_state_path, use_diff=False)
        logger.info(f"Training state is saved to {train_state_path}")
        # Save a signature to identify the best model.
        signature_path = os.path.join(save_folder, f"{reason}.signature")
        with open(signature_path, "w") as fp:
            logger.info("A signature is saved.")

    def load_saved_config(self, mode, args, logger):
        return OptionsLoader.load(mode, args, logger)

    def get_parameter_group(self, model, training_config):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": training_config.optimizer_main["weight_decay"],
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return optimizer_grouped_parameters


class AccelerateSaveLoad(SaveLoadBase):
    def __init__(self, accelerator):
        super().__init__()
        self.accelerator = accelerator

    def save(self,
            epoch_index,
            batch_index,
            args,
            options,
            model,
            optimizers,
            best_score,
            batch_count,
            train_global_count,
            folder='epoch_batch',
            reason=None,
            regex_file_pattern=None,
            logger=None):
        str_id = self.make_id(epoch_index, batch_index, folder)
        save_folder = self.get_save_dir(args=args,
                                        model_name=options.aux_model.model_name,
                                        logger=logger,
                                        str_id=str_id)
        # Save configurations
        if self.accelerator.state.local_process_index == 0:
            self.save_config_state(epoch_index=epoch_index,
                                batch_index=batch_index,
                                args=args,
                                options=options,
                                best_score=best_score,
                                batch_count=batch_count,
                                train_global_count=train_global_count,
                                save_folder=save_folder,
                                reason=reason,
                                str_id=str_id,
                                regex_file_pattern=regex_file_pattern,
                                logger=logger)

        # Save model stuff by accelerator. Different GPUs will have different random states.
        self.accelerator.save_state(save_folder)
        logger.info(f"Model is saved to folder {save_folder}")

    def load(self, args, options, model, logger):
        if options.training.train_state.model_id:
            model_path = os.path.join(get_save_load_dir(args, options.aux_model.model_name),
                                        options.training.train_state.model_id)
            try:
                self.accelerator.load_state(model_path)
                logger.info(f"Model initialised from checkpoint @ {model_path}")
            except FileNotFoundError as ex:
                logger.warning(f"Model checkpoint not found @ {model_path}")
                raise
            except IndexError as ex:
                # When trained on multi GPU but test on a single GPU,
                # the random states beyond first GPU will throw this
                # if 'accelerate config' is not reconfigured to single GPU.
                # By this stage, we have load all other states
                # except custom states which we don't have.
                # So, should be ok to preceed.
                logger.warning(ex)


import random, numpy as np
from transformers import AdamW
class SingletonSaveLoad(SaveLoadBase):
    def __init__(self, model_ext=".pth"):
        super().__init__()
        self.model_ext = model_ext

    def save(self,
            epoch_index,
            batch_index,
            args,
            options,
            model,
            optimizers,
            best_score,
            batch_count,
            train_global_count,
            folder='epoch_batch',
            reason=None,
            regex_file_pattern=None,
            logger=None):
        str_id = self.make_id(epoch_index, batch_index, folder)
        save_folder = self.get_save_dir(args=args,
                                        model_name=options.aux_model.model_name,
                                        logger=logger,
                                        str_id=str_id)
        # Save configurations
        self.save_config_state(epoch_index=epoch_index,
                            batch_index=batch_index,
                            args=args,
                            options=options,
                            best_score=best_score,
                            batch_count=batch_count,
                            train_global_count=train_global_count,
                            save_folder=save_folder,
                            reason=reason,
                            str_id=str_id,
                            regex_file_pattern=regex_file_pattern,
                            logger=logger)

        #== Save model states ==#
        saved_modules = []
        # Add baseline module states
        saved_baseline_modules = ['seq2seq_state_dict']
        model_states = {
            saved_baseline_modules[0]: model.seq2seq.state_dict()
        }
        saved_modules += saved_baseline_modules

        # Add configurable module states
        saved_aux_modules = ['enctag_classifier_state_dict',
                             'dectag_classifier_state_dict',
                             'factum_state_dict']
        for name in saved_aux_modules:
            # Configurable model states
            pos = name.index('_state_dict')
            module_name = name[:pos]
            configured = model.aux_config is not None and \
                        module_name not in model.aux_config.exclude_modules
            if configured:
                module = eval(f"model.{module_name}", {"model": model})
                model_states[name] = module.state_dict()
                saved_modules.append(name)

        # Add optimizer states now
        optimizer_states = [optimizer.state_dict() for optimizer in optimizers]
        model_states['optimizer_state_dict'] = optimizer_states

        # Save
        # Model name
        _, config_name = os.path.split(options.saveload.baseline_option)
        name_part, ext_part = os.path.splitext(config_name)
        name_part = extract_base_filename(name_part, regex_file_pattern)
        model_name = name_part+'_model'+str_id+self.model_ext
        save_path = os.path.join(save_folder, model_name)
        torch.save(model_states, save_path)
        logger.info(saved_modules)
        logger.info(f"Model is saved to folder {save_folder}")

        # Save random states
        self.save_random_state(save_folder, logger)

    def save_random_state(self, save_folder, logger):
        # Random states
        # Random number generator states
        states = {}
        states_name = f"{RNG_STATE_NAME}.pkl"
        states["random_state"] = random.getstate()
        states["numpy_random_seed"] = np.random.get_state()
        states["torch_manual_seed"] = torch.get_rng_state()
        states["torch_cuda_manual_seed"] = torch.cuda.get_rng_state_all()
        output_states_file = os.path.join(save_folder, states_name)
        torch.save(states, output_states_file)
        logger.info(f"Random states saved in {output_states_file}")

    def load(self, args, options, model, logger):
        checkpoint = None
        if options.training.train_state.model_id:
            model_path = os.path.join(get_save_load_dir(args, options.aux_model.model_name),
                                        options.training.train_state.model_id,
                                        options.saveload.model_option)
            checkpoint = torch.load(model_path)
            logger.info(f"Checkpoint is reloaded from: {model_path}")
        if checkpoint:
            logger.info('Start initiating model from checkpoint')
            # Baseline module
            loaded_modules = ['seq2seq_state_dict']
            model.seq2seq.load_state_dict(checkpoint[loaded_modules[0]])

            # Auxiliary modules
            loaded_aux_modules = ['enctag_classifier_state_dict',
                                'dectag_classifier_state_dict',
                                'factum_state_dict']
            for name in loaded_aux_modules:
                # Configurable model states
                pos = name.index('_state_dict')
                module_name = name[:pos]
                configured = model.aux_config is not None and \
                            module_name not in model.aux_config.exclude_modules
                if name in checkpoint and configured:
                    module = eval(f"model.{module_name}", {"model": model})
                    module.load_state_dict(checkpoint[name])

            logger.info(loaded_modules+loaded_aux_modules)
            logger.info("Model is checkpoint initialised")

        # Optimizer
        grouped_params = self.get_parameter_group(model, options.training)
        optimizer = self.setup_single_optimizers(model,
                                    grouped_params=grouped_params,
                                    aux_cfg=options.aux_model,
                                    training_cfg=options.training,
                                    checkpoint=checkpoint)
        if isinstance(optimizer, list):
            optimizer = optimizer[0]

        # Restore random state
        load_folder = self.get_save_dir(options=options,
                                         logger=logger)
        self.load_random_state(load_folder, logger)

    def setup_single_optimizers(model, grouped_params,
                                aux_cfg, training_cfg,
                                checkpoint=None):
        optimizers = []
        if grouped_params is None:
            params = list(model.seq2seq.parameters())
            loaded_aux_modules = ['enctag_classifier_state_dict',
                                'dectag_classifier_state_dict',
                                'factum_state_dict']
            for name in loaded_aux_modules:
                # Configurable model states
                pos = name.index('_state_dict')
                module_name = name[:pos]
                configured = model.aux_config is not None and \
                            module_name not in model.aux_config.exclude_modules
                if configured:
                    module = eval(f"model.{module_name}", {"model": model})
                    params += list(module.parameters())
        else:
            params = grouped_params
        optim_cfg = training_cfg.optimizer_main
        # main_optimizer = eval('torch.optim.' + optim_cfg["optimizer"])
        main_optimizer = eval(optim_cfg["optimizer"])
        optimizers.append(main_optimizer(params,
                                        lr=optim_cfg["lr"]))

        # Load checkpoint
        if checkpoint:
            opt_ckpts = checkpoint['optimizer_state_dict']
            assert len(opt_ckpts) == len(optimizers)
            try:
                [optimizer.load_state_dict(opt_ckpts[i]) \
                    for i, optimizer in enumerate(optimizers)]
            except:
                # Module inclusion/exclusion between runs may have changed and resulted in
                # the inconsistence with saved optimizer states. But move on.
                pass
        return optimizers

    def load_random_state(self, input_dir, logger):
        # Random states
        states_name = f"{RNG_STATE_NAME}.pkl"
        states = torch.load(os.path.join(input_dir, states_name))
        random.setstate(states["random_state"])
        np.random.set_state(states["numpy_random_seed"])
        torch.set_rng_state(states["torch_manual_seed"])
        torch.cuda.set_rng_state_all(states["torch_cuda_manual_seed"])
        # ^^ safe to call this function even if cuda is not available
        logger.info("All random states loaded successfully")
