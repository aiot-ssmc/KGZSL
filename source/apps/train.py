import utils
from apps.config.data import get_dataloader, data_config
from apps.config.nn import config, model_save_dir
from apps.config.gpu import tlog

log = utils.log.get_logger()
progressbar = tlog.progressbar

hp = config.hp

best_model_dir = model_save_dir.parent / "best"
best_model_dir.mkdir()


def evaluate_now(epoch):
    return epoch % 5 == 0


class Trainer:
    def __init__(self):
        if data_config.zsl:
            from apps.zsl_model import Model
        else:
            from apps.model import Model
        self.model = Model()

        self.train_loader, self.val_loader, self.test_loader = get_dataloader(batch_size=hp.batch_size)

    def fit(self):
        max_metric = 0.0
        sub_train_loader = utils.iter.SubIter(self.train_loader, 1000 // self.train_loader.batch_size)
        # train_loader_use = utils.iter.SubIter(train_loader, 200)
        train_loader_use = self.train_loader
        eval_loader_use = self.val_loader
        test_loader_use = self.test_loader
        for epoch_num in progressbar(range(hp.total_epochs), desc="Epoch"):
            self.model.training_epoch(progressbar(train_loader_use, desc="training"), ds_name="training")
            self.model.save_model(model_dir=model_save_dir)

            # if not evaluate_now(epoch_num):
            #     continue

            metric_train = self.model.eval_epoch(progressbar(sub_train_loader, desc='evaluating train'),
                                                 ds_name="train", enhance=False)
            metric_eval = self.model.eval_epoch(progressbar(eval_loader_use, desc='evaluating'),
                                                ds_name="eval", enhance=True)
            if metric_eval > max_metric:
                max_metric = metric_eval
                metric_test = self.model.eval_epoch(progressbar(test_loader_use, desc='testing'),
                                                    ds_name="test", enhance=True)
                log.info(f"updating best model, metric_eval: {metric_eval}, metric_test: {metric_test}")
                (model_dir := best_model_dir / f"{epoch_num:03d}-eval{metric_eval:.3f}-test{metric_test:.3f}").mkdir()
                self.model.save_model(model_dir=model_dir)
            else:
                max_metric = 0.9 * max_metric + 0.1 * metric_eval

    def eval(self):
        log.info("evaluating model...")
        for _ in progressbar(range(1)):
            metric_test = self.model.eval_epoch(progressbar(self.test_loader, desc='testing'),
                                                ds_name="test", enhance=True)
            metric_eval = self.model.eval_epoch(progressbar(self.val_loader, desc='evaluating'),
                                                ds_name="eval", enhance=True)
            metric_train = self.model.eval_epoch(progressbar(self.train_loader, desc='evaluating train'),
                                                 ds_name="train", enhance=False)

            log.info(f"metric_test: {metric_test}, metric_eval: {metric_eval}, metric_train: {metric_train}")


def main():
    trainer = Trainer()
    trainer.fit()
    trainer.eval()


if __name__ == '__main__':
    main()
