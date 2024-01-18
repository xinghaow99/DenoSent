from transformers import Trainer
from transformers.trainer import unwrap_model
from typing import List, Optional, Dict
import wandb

import sys

from torch.utils.data.dataset import Dataset

from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES



# Set path to SentEval
PATH_TO_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data'

# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

from mteb import MTEB

class MyTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_stsb_spearman = 0
        self.best_sickr_spearman = 0
        self.best_avg_sts = 0

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        metrics = {}
        # SentEval prepare and batcher
        def prepare(params, samples):
            return

        def batcher(params, batch):
            sentences = [' '.join(s) for s in batch]
            return self.model.encode(sentences, len(sentences))

        # Set params for SentEval (fastmode)
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
        params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                            'tenacity': 3, 'epoch_size': 2}

        se = senteval.engine.SE(params, batcher, prepare)
        tasks = ['STSBenchmark', 'SICKRelatedness']
        self.model.eval()
        results = se.eval(tasks)

        stsb_spearman = results['STSBenchmark']['dev']['spearman'][0]
        sickr_spearman = results['SICKRelatedness']['dev']['spearman'][0]
        # evaluation = MTEB(tasks=['STSBenchmark'], task_langs=["en"], task_categories=['S2S'])
        # results = evaluation.run(self.model, verbosity=0, output_folder=None, eval_splits=['validation'], batch_size=self.args.eval_batch_size)
        # stsb_spearman = results['STSBenchmark']['validation']['cos_sim']['spearman']
        # sickr_spearman = results['SICK-R']['validation']['cos_sim']['spearman']
        metrics.update({"eval_stsb_spearman": stsb_spearman, "eval_sickr_spearman": sickr_spearman, "eval_avg_sts": (stsb_spearman + sickr_spearman) / 2})
        # metrics.update({"eval_stsb_spearman": stsb_spearman})
        if stsb_spearman > self.best_stsb_spearman:
            self.best_stsb_spearman = stsb_spearman
        if sickr_spearman > self.best_sickr_spearman:
            self.best_sickr_spearman = sickr_spearman
        if (stsb_spearman + sickr_spearman) / 2 > self.best_avg_sts:
            self.best_avg_sts = (stsb_spearman + sickr_spearman) / 2
        wandb.run.summary["best_stsb_spearman"] = self.best_stsb_spearman
        wandb.run.summary["best_sickr_spearman"] = self.best_sickr_spearman
        wandb.run.summary["best_avg_sts"] = self.best_avg_sts
        self.log(metrics)
        return metrics

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs, global_step=self.state.global_step, max_steps=self.state.max_steps)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss
