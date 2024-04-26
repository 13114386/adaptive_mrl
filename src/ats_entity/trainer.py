from __future__ import unicode_literals, print_function, division
'''
    Refactor from https://github.com/huggingface/transformers/blob/master/examples/pytorch/summarization/run_summarization_no_trainer.py
'''
import torch
from common.metric_meter import MetricMeter, ProgressCounter
from common.timer_deco import timerdeco
from common.ml_except import EarlyStopException
from common.earlystop_cond import EarlyStopTimeLimitCondition, EarlyStopConditionByCount
from checkpoint import get_start_point

class Trainer():
    def __init__(self, validator, checkpoint, skip_except=False):
        self.validator = validator
        self.checkpoint = checkpoint
        self.skip_except = skip_except

    def train_per_iteration(
        self,
        iepoch,
        ibatch,
        options,
        model,
        optimizer,
        lr_scheduler,
        batch,
        accelerator,
        train_glb_counter,
        avg_cost_mtr,
        avg_x_tag_acc_mtr,
        avg_y_tag_acc_mtr,
        avg_y_inarc_acc_mtr,
        n_train_iterations,
        log_freq,
        summary_writer,
        logger
    ):
        '''
            Gigaword dataset may have some coreference without endophora and cause exception.
            Skip the training iteration on such ill-formed batch.
            Refer to https://discuss.pytorch.org/t/how-to-try-catch-errors-during-training/108619
            for try...except... setup during training.
        '''
        try:
            model.train()
            outputs = model(batch[:-1], options=options, iepoch=iepoch)
            x_tag_acc = outputs.get("x_tag_acc", torch.tensor(0.0)).item()
            y_tag_acc = outputs.get("y_tag_acc", torch.tensor(0.0)).item()
            y_inarc_acc = outputs.get("y_inarc_acc", torch.tensor(0.0)).item()
            avg_x_tag_acc_mtr(x_tag_acc)
            avg_y_tag_acc_mtr(y_tag_acc)
            avg_y_inarc_acc_mtr(y_inarc_acc)

            loss = outputs["cost"]
            loss_data = loss.item()
            avg_cost_mtr(loss_data)
            train_glb_counter += 1
            if summary_writer:
                summary_writer.add_scalar("Loss/train", loss_data, train_glb_counter.count)
                summary_writer.add_scalar("Avg loss/train", avg_cost_mtr.average(), train_glb_counter.count)
                summary_writer.add_scalar("Acc (x POS tag)/train", x_tag_acc, train_glb_counter.count)
                summary_writer.add_scalar("Avg acc (x POS tag)/train", avg_x_tag_acc_mtr.average(), train_glb_counter.count)
                summary_writer.add_scalar("Acc (y POS tag)/train", y_tag_acc, train_glb_counter.count)
                summary_writer.add_scalar("Avg acc (y POS tag)/train", avg_y_tag_acc_mtr.average(), train_glb_counter.count)
                summary_writer.add_scalar("Acc (y inarc)/train", y_inarc_acc, train_glb_counter.count)
                summary_writer.add_scalar("Avg acc (y inarc)/train", avg_y_inarc_acc_mtr.average(), train_glb_counter.count)

            if (ibatch+1) % log_freq == 0:
                logger.info((f'Train: Epoch {iepoch}, iBatch {ibatch}: '
                            f'Cost {loss_data}, AvgCost {avg_cost_mtr.average()}, '
                            f'AvgXTagAcc {avg_x_tag_acc_mtr.average()}, xTagAcc {x_tag_acc}, '
                            f'AvgYTagAcc {avg_y_tag_acc_mtr.average()}, yTagAcc {y_tag_acc}, '
                            f'AvgYInArcAcc {avg_y_inarc_acc_mtr.average()}, yInArcAcc {y_inarc_acc}'))

            # Normalize loss to account for batch accumulation.
            loss = loss / options.training.gradient_accumulation_steps

            accelerator.backward(loss)

            # Weights update on gradient accumulation.
            if (ibatch+1) % options.training.gradient_accumulation_steps == 0 or \
                ibatch == n_train_iterations - 1:
                optimizer.step()
                lr_scheduler.step()
                model.zero_grad(set_to_none=True)
                # progress_bar.update(1)
                # Clip weights.
                if options.training.grad_clip > 0 and \
                    train_glb_counter.count % options.training.grad_clip_freq == 0:
                        accelerator.clip_grad_value_(model.parameters(), options.training.grad_clip)
            skip_batch = False
        except Exception as e:
            data_idxs = batch[-1]
            logger.warning(f'Training caugh an error: {str(e)} with batch data indexes ({data_idxs})')
            if self.skip_except:
                skip_batch = True
            else:
                raise
            loss_data, x_tag_acc, y_tag_acc, y_inarc_acc = 0.0, 0.0, 0.0, 0.0
        return (loss_data, x_tag_acc, y_tag_acc, y_inarc_acc), skip_batch

    @timerdeco("epoch")
    def train_epoch(
        self,
        iepoch,
        start_iteration,
        options,
        train_dataloader,
        val_dataloader,
        model,
        optimizer,
        lr_scheduler,
        accelerator,
        max_steps_stop,
        avg_cost_mtr,
        avg_x_tag_acc_mtr,
        avg_y_tag_acc_mtr,
        avg_y_inarc_acc_mtr,
        train_glb_counter,
        log_freq,
        summary_writer,
        logger
    ):
        '''
            train epoch
        '''
        early_stop = False
        n_train_iterations = options.training.n_train_iterations
        for ibatch, batch in zip(range(start_iteration, n_train_iterations), train_dataloader):
            metrics, skip_batch = \
                self.train_per_iteration(iepoch=iepoch,
                                        ibatch=ibatch,
                                        options=options,
                                        model=model,
                                        optimizer=optimizer,
                                        lr_scheduler=lr_scheduler,
                                        batch=batch,
                                        accelerator=accelerator,
                                        train_glb_counter=train_glb_counter,
                                        avg_cost_mtr=avg_cost_mtr,
                                        avg_x_tag_acc_mtr=avg_x_tag_acc_mtr,
                                        avg_y_tag_acc_mtr=avg_y_tag_acc_mtr,
                                        avg_y_inarc_acc_mtr=avg_y_inarc_acc_mtr,
                                        n_train_iterations=n_train_iterations,
                                        log_freq=log_freq,
                                        summary_writer=summary_writer,
                                        logger=logger)
            loss_data, x_tag_acc, y_tag_acc, y_inarc_acc = metrics

            if skip_batch:
                continue

            if (ibatch+1) % options.training.gradient_accumulation_steps == 0 or \
                ibatch == n_train_iterations - 1:
                # Stop check.
                max_steps_stop.incr()
                early_stop = max_steps_stop()
                if early_stop:
                    break

        if (ibatch+1) % log_freq != 0: # Log remaning batch results
            logger.info((f'Train: Epoch {iepoch}, iBatch {ibatch}: '
                        f'Cost {loss_data}, AvgCost {avg_cost_mtr.average()}, '
                        f'xTagAcc {x_tag_acc}, AvgXTagAcc {avg_x_tag_acc_mtr.average()}, '
                        f'yTagAcc {y_tag_acc}, AvgYTagAcc {avg_y_tag_acc_mtr.average()}, '
                        f'yInArcAcc {y_inarc_acc}, AvgYInArcAcc {avg_y_inarc_acc_mtr.average()}'))

        try:
            # Run regular validation per epoch
            self.validator(iepoch=iepoch,
                            ibatch=ibatch,
                            options=options,
                            val_dataloader=val_dataloader,
                            model=model,
                            accelerator=accelerator,
                            glb_count=train_glb_counter.count,
                            summary_writer=summary_writer,
                            logger=logger)
        except EarlyStopException as ex:
            logger.warning(str(ex))
            early_stop = True

        if summary_writer:
            summary_writer.flush()
        return ibatch, early_stop

    @timerdeco("session")
    def __call__(
        self,
        args,
        options,
        datasets,
        model,
        optimizer,
        lr_scheduler,
        accelerator,
        max_train_steps,
        summary_writer,
        logger
    ):
        avg_x_tag_acc_mtr = MetricMeter()
        avg_y_tag_acc_mtr = MetricMeter()
        avg_y_inarc_acc_mtr = MetricMeter()
        avg_cost_mtr = MetricMeter(0.95)
        train_dataloader, val_dataloader = datasets
        if args.dataset_changed:
            (start_epoch, start_iteration) = (options.training.train_state.start_epoch, 0)
        else:
            (start_epoch, start_iteration) = get_start_point(options.training.n_train_iterations,
                                                            options.training.train_state.train_global_count)
        logger.info(f"start_epoch, start_iteration: ({start_epoch}, {start_iteration})")

        if start_epoch >= options.training.max_epochs:
            logger.warning(f"Start epoch {start_epoch} should be less than "
                           f"max epochs {options.training.max_epochs}")
            return

        # progress_bar = tqdm(range(args.max_train_steps),
        #                   disable=not accelerator.is_local_main_process)
        timelimit_cond = EarlyStopTimeLimitCondition(time_limit=args.time_limit)
        max_steps_stop = EarlyStopConditionByCount(max_train_steps)
        train_glb_counter = ProgressCounter(options.training.train_state.train_global_count)
        time_elapser = timerdeco(verbose=False)
        for iepoch in range(start_epoch, options.training.max_epochs):
            logger.info(f'Epoch {iepoch}')
            with time_elapser:
                torch.cuda.empty_cache()
                ibatch, early_stop = \
                    self.train_epoch(iepoch=iepoch,
                                    start_iteration=start_iteration,
                                    options=options,
                                    train_dataloader=train_dataloader,
                                    val_dataloader=val_dataloader,
                                    model=model,
                                    optimizer=optimizer,
                                    lr_scheduler=lr_scheduler,
                                    accelerator=accelerator,
                                    max_steps_stop=max_steps_stop,
                                    avg_cost_mtr=avg_cost_mtr,
                                    avg_x_tag_acc_mtr=avg_x_tag_acc_mtr,
                                    avg_y_tag_acc_mtr=avg_y_tag_acc_mtr,
                                    avg_y_inarc_acc_mtr=avg_y_inarc_acc_mtr,
                                    train_glb_counter=train_glb_counter,
                                    log_freq=args.log_freq,
                                    summary_writer=summary_writer,
                                    logger=logger)

            early_stop = timelimit_cond(time_elapser.elapsed) or early_stop

            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            reason = "last" if early_stop or (iepoch+1==options.training.max_epochs) else "regular"
            early_stop_2 = self.checkpoint(iepoch=iepoch,
                                        ibatch=ibatch,
                                        args=args,
                                        options=options,
                                        train_global_count=train_glb_counter.count,
                                        score=avg_cost_mtr.average(),
                                        model=unwrapped_model,
                                        optimizers=[optimizer],
                                        accelerator=accelerator,
                                        reason=reason,
                                        logger=logger)
            early_stop = early_stop or early_stop_2
            # Reset start iteration.
            start_iteration = 0

            if early_stop:
                break

        logger.info(f"Training completes @ epoch {iepoch}.")
