"""
### 참고 자료

##### 1. 논문
   - [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
   - [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

##### 2. 코드
   - [Text classification examples (transformers)](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification)
   - [Image classification examples (lightning)](https://github.com/Lightning-AI/lightning/tree/master/examples/fabric/image_classifier)
   - [Finetuning (KoELECTRA)](https://github.com/monologg/KoELECTRA/tree/master/finetune)
   - [Process (datasets)](https://huggingface.co/docs/datasets/process)
"""
from __future__ import annotations

import evaluate
from .finetuner import MyFinetuner, HeadModel, MongoClient
from ..common.util import *


class MyPredictor(MyFinetuner):
    """
    Predictor for sentence-level classification or regression tasks.
    - Refer to `lightning.fabric.Fabric`
    """

    def __init__(self, *args, **kwargs):
        super(MyPredictor, self).__init__(*args, **kwargs)

    def run(self):
        with MyTimer(f"Predicting({self.state.data_name}/{self.state.data_part})", prefix=self.prefix, postfix=self.postfix, mb=1, rt=1, rb=1, rc='=', verbose=self.is_global_zero):
            # BEGIN
            with MyTimer(verbose=self.is_global_zero):
                self.show_state_values(verbose=self.is_global_zero)
                assert self.state.data_name and isinstance(self.state.data_name, (Path, str)), f"Invalid data_name: ({type(self.state.data_name).__qualname__}) {self.state.data_name}"

            # READY(data)
            with MyTimer(verbose=self.is_global_zero):
                self.prepare_datasets(verbose=self.is_global_zero)
                self.prepare_dataloader()

            # READY(finetuning)
            self.finetuning_model = HeadModel(state=self.state, tokenizer=self.tokenizer)
            with MyTimer(verbose=self.is_global_zero, rb=1):
                self.check_pretrained(sample=self.is_global_zero)

            with MyTimer(verbose=self.is_global_zero, rb=1 if self.state.strategy == 'deepspeed' else 0):
                self.optimizer = self.configure_optimizer()
                self.scheduler = self.configure_scheduler()
                self.loss_metric = self.configure_loss()
                self.score_metric = evaluate.load(self.state.score_metric.major, self.state.score_metric.minor)
                self.state['finetuning_model'] = f"{type(self.finetuning_model).__qualname__} | pretrained={self.state.pretrained.path.name}"
                self.state['optimizer'] = f"{type(self.optimizer).__qualname__} | lr={self.state.learning_rate}"
                self.state['scheduler'] = f"{type(self.scheduler).__qualname__} | gamma={self.state.lr_scheduler_gamma}"
                self.state['loss_metric'] = f"{type(self.loss_metric).__qualname__}"
                self.state['score_metric'] = f"{self.state.score_metric.major}/{self.state.score_metric.minor}"
                for k in ('finetuning_model', 'optimizer', 'scheduler', 'loss_metric', 'score_metric'):
                    print(f"- {k:30s} = {self.state[k]}")
            with MyTimer(verbose=self.is_global_zero, rb=1):
                self.finetuning_model, self.optimizer = self.setup(self.finetuning_model, self.optimizer)

            # READY(predicting)
            assert self.state.finetuned_home and isinstance(self.state.finetuned_home, Path), f"Invalid finetuned_home: ({type(self.state.finetuned_home).__qualname__}) {self.state.finetuned_home}"
            assert self.state.finetuned_sub and isinstance(self.state.finetuned_sub, (Path, str)), f"Invalid finetuned_sub: ({type(self.state.finetuned_sub).__qualname__}) {self.state.finetuned_sub}"
            records_to_predict = []
            with MyTimer(verbose=self.is_global_zero, rb=1):
                finetuned_dir: Path = self.state.finetuned_home / self.state.data_name / self.state.finetuned_sub
                assert finetuned_dir and finetuned_dir.is_dir(), f"Invalid finetuned_dir: ({type(finetuned_dir).__qualname__}) {finetuned_dir}"
                finetuner_state_path: Path or None = exists_or(finetuned_dir / 'finetuner_state.json')
                assert finetuner_state_path and finetuner_state_path.is_file(), f"Invalid finetuner_state_path: ({type(finetuned_dir / 'finetuner_state.json').__qualname__}) {finetuned_dir / 'finetuner_state.json'}"
                finetuner_state: AttrDict = load_attrs(finetuner_state_path)
                for record in finetuner_state.records:
                    record.model_path = finetuned_dir / Path(record.model_path).name
                    to_predict = len(records_to_predict) <= (self.state.num_train_epochs - 1) and record.model_path and record.model_path.exists()
                    print(f"- [{'O' if record.model_path.exists() else 'X'}] {record.model_path} => [{'O' if to_predict else 'X'}]")
                    if to_predict:
                        records_to_predict.append(record)
            with MyTimer(verbose=self.is_global_zero, rb=1, rc='='):
                print(f"- {'finetuned_home':30s} = {self.state.finetuned_home}")
                print(f"- {'finetuned_sub':30s} = {self.state.finetuned_sub}")
                print(f"- {'finetuned.num_records':30s} = {len(finetuner_state.records)}")
                print(f"- {'finetuned.num_train_epochs':30s} = {finetuner_state.num_train_epochs}")
                print(f"- {'predicted.num_train_epochs':30s} = {self.state.num_train_epochs}")
                print(f"- {'predicted.target_model_names':30s} = {', '.join(r.model_path.name for r in records_to_predict)}")
                print(f"- {'predicted_home':30s} = {self.state.predicted_home}")
                print(f"- {'predicted_sub':30s} = {self.state.predicted_sub}")

            # READY(output)
            assert self.state.predicted_home and isinstance(self.state.predicted_home, Path), f"Invalid predicted_home: ({type(self.state.predicted_home).__qualname__}) {self.state.predicted_home}"
            assert isinstance(self.state.predicted_sub, (type(None), Path, str)), f"Invalid predicted_sub: ({type(self.state.predicted_sub).__qualname__}) {self.state.predicted_sub}"
            tab_name: str = self.state.predicted_sub if self.state.predicted_sub else "default"
            predicted_dir: Path = make_dir(self.state.predicted_home / self.state.data_name / tab_name)
            predicted_files = {
                "state": predicted_dir / "predictor_state.json",
                "preds": predicted_dir / "predict.tsv",
            }
            self.state["records"] = list()

            # EPOCH
            with StageMarker(self.global_rank, self.world_size, self.milestones, db_name=self.state.data_name, tab_name=tab_name, host=self.db_host, port=self.db_port) as marker:
                for i, record in enumerate(records_to_predict):
                    with MyTimer(verbose=True, rb=1 if self.is_global_zero and i < len(records_to_predict) - 1 else 0, flush_sec=0.3):
                        # INIT
                        metrics = {}
                        predict = {}
                        current = f"(Epoch {record.epoch:02.0f})"
                        marker.initialize(stage=current)
                        marker.mark_done("INIT", stage=current)
                        with MyTimer(verbose=True, flush_sec=0.3):
                            print(self.time_tqdm.to_desc(pre=current, desc=f"composed #{self.global_rank + 1:01d}") + f": model | {record.model_path}")

                        # LOAD
                        assert not any(c in str(record.model_path) for c in ['*', '?', '[', ']']), f"Invalid model path: {record.model_path}"
                        model_state_dict = self.load(record.model_path)
                        self.finetuning_model.load_state_dict(model_state_dict, strict=False)
                        marker.mark_done("LOAD", stage=current)
                        with MyTimer(verbose=True, flush_sec=0.3):
                            if self.is_global_zero and "metrics" in record:
                                for name, score in record.metrics.items():
                                    print(self.time_tqdm.to_desc(pre=current, desc=f"reported as") +
                                          f": {name:<5s} | {', '.join(f'{k}={score[k]:.4f}' for k in append_intersection(score.keys(), ['runtime']))}")

                        # APPLY
                        self.finetuning_model.eval()
                        with torch.no_grad():
                            for k in self.input_datasets.keys():
                                if k not in self.dataloader or not self.dataloader[k]:
                                    continue
                                if k not in self.state.predicting_splits or not self.state.predicting_splits[k]:
                                    continue
                                inputs = []
                                outputs = []
                                dataloader = self.dataloader[k]
                                with MyTimer(flush_sec=0.3) as timer:
                                    tqdm = self.time_tqdm if self.is_global_zero else self.mute_tqdm
                                    for batch_idx, batch in enumerate(
                                            tqdm(dataloader, position=self.global_rank,
                                                 pre=current, desc=f"metering #{self.global_rank + 1:01d}", unit=f"x{dataloader.batch_size}")):
                                        batch = to_tensor_batch(batch, input_keys=self.tokenizer.model_input_names)
                                        output = self.each_step(batch, batch_idx, input_keys=self.tokenizer.model_input_names)
                                        outputs.append(output)
                                        inputs.append(batch)
                                metrics[k] = self.outputs_to_metrics(outputs, timer=timer)
                                predict[k] = self.outputs_to_predict(outputs, inputs=inputs, with_label=False)
                        marker.mark_done("APPLY", stage=current)
                        with MyTimer(verbose=True, flush_sec=0.3):
                            for name, score in metrics.items():
                                print(self.time_tqdm.to_desc(pre=current, desc=f"measured #{self.global_rank + 1:01d}") +
                                      f": {name:<5s} | {', '.join(f'{k}={score[k]:.4f}' for k in append_intersection(score.keys(), ['runtime']))}")

                        # SAVE
                        if self.state.predicted_sub:
                            for k in self.input_datasets.keys():
                                if k not in self.dataloader or not self.dataloader[k]:
                                    continue
                                if k not in self.state.predicting_splits or not self.state.predicting_splits[k]:
                                    continue
                                if len(predict[k]) <= 0:
                                    continue
                                data_prefix = Path(self.state.data_files[k]).parent.stem
                                preds_path = new_path(new_path(predicted_files["preds"], post=f'{record.epoch:02.0f}e'), pre=data_prefix, sep='=')
                                state_path = new_path(predicted_files["state"], pre=data_prefix, sep='=')
                                if self.is_global_zero:
                                    record["metrics"][f"preds-{k}"] = metrics[k]
                                    record["preds_path"] = preds_path
                                    self.state["records"].append(record)
                                    save_attrs(self.state, file=state_path, keys=self.state.log_targets)
                                save_rows(predict[k], file=preds_path, with_column_name=True)
                                if preds_path.exists():
                                    print(self.time_tqdm.to_desc(pre=current, desc=f"exported #{self.global_rank + 1:01d}") + f": preds | {preds_path}")
                            marker.mark_done("SAVE", stage=current)

    @staticmethod
    def outputs_to_predict(outputs, inputs, with_label=False):
        cols = {}
        for k in inputs[0].keys():
            cols[k] = list(chain.from_iterable(batch[k] for batch in inputs))
        cols['predict'] = torch.cat([x['p'] for x in outputs]).detach().cpu().numpy().tolist()
        if with_label:
            cols['label'] = torch.cat([x['y'] for x in outputs]).detach().cpu().numpy().tolist()
        rows = []
        for i in range(len(cols['predict'])):
            rows.append({k: cols[k][i] for k in cols.keys()})
        return rows
