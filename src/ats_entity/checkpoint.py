from __future__ import unicode_literals, print_function, division


def get_start_point(n_iterations, global_count):
    start_epoch = global_count // n_iterations
    start_iteration = global_count % n_iterations
    return (start_epoch, start_iteration)

class Checkpoint():
    def __init__(self, training_state, regex_file_pattern, save_load):
        self.best_score = training_state.best_score
        self.batch_count = training_state.batch_count
        self.regex_file_pattern = regex_file_pattern
        self.save_load = save_load

    def __call__(
        self,
        iepoch,
        ibatch,
        args,
        options,
        train_global_count,
        score,
        model,
        optimizers,
        accelerator,
        reason,
        logger
    ):
        '''
            Note that:
                Under multi-GPU setting, Model on one GPU may get the best result
                while the replicated model on the other GPU may not have the best result.
                Under such circumstance, the under-performed GPU will not save its model
                and its configuration settings if it is the main process.
                There is not a workaround at the moment since synchronizing both is non-trivial.
                However, regular saving guarantees on all GPUs since they are on the same
                iteration and epoch.
                Besides, the best performce on all GPUs at the same iteration can happen
                and models on all GPUs will be saved.
        '''
        early_stop = False
        logger.info(f'New score is {score}, best score so far is {self.best_score}')
        # Learning rates
        lrs = [param_group['lr'] for optimizer in optimizers \
                for param_group in optimizer.param_groups]
        fmt = ', '.join('%.15f' % l for l in lrs)
        logger.info(f'Current learning rate is/are {fmt}')
        delta_count = train_global_count - self.batch_count
        if score < self.best_score:
            logger.info('Update best model')
            self.best_score = score
            self.batch_count += delta_count
            self.save_load.save(epoch_index=iepoch,
                                batch_index=ibatch,
                                args=args,
                                options=options,
                                model=model,
                                optimizers=optimizers,
                                best_score=self.best_score,
                                batch_count=self.batch_count,
                                train_global_count=train_global_count,
                                folder='epoch_batch',
                                reason='best',
                                regex_file_pattern=self.regex_file_pattern,
                                logger=logger)
        else:
            if (reason == "regular" and (iepoch+1) % options.training.save_mode["freq"] == 0) or \
                (reason == "last" and (iepoch+1) % options.training.save_mode["freq"] != 0):
                logger.info('Update model on epoch')
                self.save_load.save(epoch_index=iepoch,
                                    batch_index=ibatch,
                                    args=args,
                                    options=options,
                                    model=model,
                                    optimizers=optimizers,
                                    best_score=self.best_score,
                                    batch_count=self.batch_count,
                                    train_global_count=train_global_count,
                                    folder='epoch_batch',
                                    reason=reason,
                                    regex_file_pattern=self.regex_file_pattern,
                                    logger=logger)

            early_stop = options.training.earlystop.on and \
                        delta_count > options.training.earlystop.stop_bound
        return early_stop
