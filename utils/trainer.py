import os
import time
import shutil
import torch
from utils.metrics_manager import MetricsManager
from tensorboardX import SummaryWriter


class Trainer(object):
    """
    Trainer
    """
    def __init__(
            self,
            model,
            optimizer,
            train_iter,
            valid_iter,
            logger,
            valid_metric_name="-loss",
            num_epochs=1,
            save_dir=None,
            log_steps=None,
            valid_steps=None,
            grad_clip=None,
            lr_scheduler=None,
            save_summary=False
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_iter = train_iter
        self.valid_iter = valid_iter
        self.logger = logger

        self.is_decreased_valid_metric = valid_metric_name[0] == "-"
        self.valid_metric_name = valid_metric_name[1:]
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        self.log_steps = log_steps
        self.valid_steps = valid_steps
        self.grad_clip = grad_clip
        self.lr_scheduler = lr_scheduler
        self.save_summary = save_summary

        if self.save_summary:
            self.train_writer = SummaryWriter(
                os.path.join(self.save_dir, "logs", "train"))
            self.valid_writer = SummaryWriter(
                os.path.join(self.save_dir, "logs", "valid"))

        self.best_valid_metric = float(
            "inf") if self.is_decreased_valid_metric else -float("inf")
        self.epoch = 0
        self.best_epoch = 0
        self.batch_num = 0

        self.train_start_message = "\n".join(["",
                                              "=" * 85,
                                              "=" * 34 + " Model Training " + "=" * 35,
                                              "=" * 85,
                                              ""])
        self.valid_start_message = "\n" + "-" * 33 + " Model Evaluation " + "-" * 33

    def summarize_train_metrics(self, metrics, global_step):
        """
        summarize_train_metrics
        """
        for key, val in metrics.items():
            if isinstance(val, (list, tuple)):
                val = val[0]
            if isinstance(val, torch.Tensor):
                self.train_writer.add_scalar(key, val, global_step)

    def summarize_valid_metrics(self, metrics_mm, global_step):
        """
        summarize_valid_metrics
        """
        for key in metrics_mm.metrics_cum.keys():
            val = metrics_mm.get(key)
            self.valid_writer.add_scalar(key, val, global_step)

    def train_epoch(self):
        """
        A training epoch, saves model parameters every `valid_steps`
        """
        self.epoch += 1
        train_mm = MetricsManager()
        num_batches = len(self.train_iter)
        self.logger.info(self.train_start_message)

        start_time = time.time()
        for batch_id, inputs in enumerate(self.train_iter, 1):
            self.model.train()
            # Do a training iteration
            metrics = self.model.iterate(
                inputs,
                optimizer=self.optimizer,
                grad_clip=self.grad_clip,
            )

            train_mm.update(metrics)
            self.batch_num += 1

            if batch_id % self.log_steps == 0:
                elapsed = time.time() - start_time
                message_prefix = "[Train][{:2d}][{}/{}]".format(self.epoch, batch_id, num_batches)
                metrics_message = train_mm.report_cum()
                train_mm.clear()
                message_posfix = "TIME-{:.2f}".format(elapsed)
                self.logger.info("   ".join(
                    [message_prefix, metrics_message, message_posfix]))
                if self.save_summary:
                    self.summarize_train_metrics(metrics, self.batch_num)
                start_time = time.time()

            if batch_id % self.valid_steps == 0:
                self.logger.info(self.valid_start_message)
                start_time = time.time()
                valid_mm = self.evaluate()
                elapsed = time.time() - start_time

                message_prefix = "[Valid][{:2d}][{}/{}]".format(self.epoch, batch_id, num_batches)
                message_posfix = "TIME-{:.2f}".format(elapsed)
                metrics_message = valid_mm.report_cum()
                self.logger.info("   ".join([message_prefix, metrics_message, message_posfix]))

                if self.save_summary:
                    self.summarize_valid_metrics(valid_mm, self.batch_num)

                cur_valid_metric = valid_mm.get(self.valid_metric_name)
                if self.is_decreased_valid_metric:
                    is_best = cur_valid_metric < self.best_valid_metric
                else:
                    is_best = cur_valid_metric > self.best_valid_metric
                if is_best:
                    self.best_valid_metric = cur_valid_metric
                self.save(is_best)
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step(cur_valid_metric)
                self.logger.info("-" * 85 + "\n")
                start_time = time.time()

        self.save()
        self.logger.info('')

    def train(self):
        """
        train
        """
        for _ in range(self.epoch, self.num_epochs):
            self.train_epoch()
        self.logger.info('Train finished!\n')
        self.logger.info('Best model state at epoch {}.\n'.format(self.best_epoch))

    def save(self, is_best=False):
        """
        save
        """
        model_file = os.path.join(
            self.save_dir, "state_epoch_{}.model".format(self.epoch))
        torch.save(self.model.state_dict(), model_file)
        self.logger.info("Saved model state to '{}'".format(model_file))

        train_file = os.path.join(
            self.save_dir, "state_epoch_{}.train".format(self.epoch))
        train_state = {"epoch": self.epoch,
                       "batch_num": self.batch_num,
                       "best_valid_metric": self.best_valid_metric,
                       "optimizer": self.optimizer.state_dict()}
        if self.lr_scheduler is not None:
            train_state["lr_scheduler"] = self.lr_scheduler.state_dict()
        torch.save(train_state, train_file)
        self.logger.info("Saved train state to '{}'".format(train_file))

        if is_best:
            self.best_epoch = self.epoch
            best_model_file = os.path.join(self.save_dir, "best.model")
            best_train_file = os.path.join(self.save_dir, "best.train")
            shutil.copy(model_file, best_model_file)
            shutil.copy(train_file, best_train_file)
            self.logger.info(
                "Saved best model state to '{}' with new best valid metric {}-{:.3f}".format(
                    best_model_file, self.valid_metric_name.upper(), self.best_valid_metric))

    def load(self, file_prefix):
        """
        load
        """
        model_file = "{}.model".format(file_prefix)
        train_file = "{}.train".format(file_prefix)

        model_state_dict = torch.load(model_file, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(model_state_dict)
        self.logger.info("Loaded model state from '{}'".format(model_file))

        train_state_dict = torch.load(train_file, map_location=lambda storage, loc: storage)
        self.epoch = train_state_dict["epoch"]
        self.best_valid_metric = train_state_dict["best_valid_metric"]
        self.batch_num = train_state_dict["batch_num"]
        self.optimizer.load_state_dict(train_state_dict["optimizer"])
        if self.lr_scheduler is not None and "lr_scheduler" in train_state_dict:
            self.lr_scheduler.load_state_dict(train_state_dict["lr_scheduler"])
        self.logger.info(
            "Loaded train state from '{}' with (epoch-{} best_valid_metric-{:.3f})".format(
                train_file, self.epoch, self.best_valid_metric))

    def evaluate(self):
        """
        evaluate
        """
        self.model.eval()
        mm = MetricsManager()
        with torch.no_grad():
            for inputs in self.valid_iter:
                metrics = self.model.iterate(inputs=inputs)
                mm.update(metrics)
        return mm
