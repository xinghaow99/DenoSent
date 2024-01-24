# DenoSent: A Denoising Objective for Self-Supervised Sentence Representation Learning

Official repo for our AAAI 2024 paper: DenoSent: A Denoising Objective for Self-Supervised Sentence Representation Learning.

## Getting Started

Run `pip install -r requirements.txt` to prepare the environment.

Use the script from the [SimCSE repo](https://github.com/princeton-nlp/SimCSE) to download the datasets for SentEval evaluation:
```
cd SentEval/data/downstream/
bash download_dataset.sh
```
 
 ## Access Our Model and Dataset from Huggingface🤗
Both our [model checkpoint](https://huggingface.co/Singhoo/denosent-bert-base) and [dataset](https://huggingface.co/datasets/Singhoo/denosent_data) are available on 🤗.

Generate embeddings with DenoSent:
 ```
 from transformers import AutoModel

model = AutoModel.from_pretrained("Singhoo/denosent-bert-base", trust_remote_code=True)

sentences = [
    "The curious cat tiptoed across the creaky wooden floor, pausing to inspect a fluttering curtain.",
    "A lone hiker stood atop the misty mountain, marveling at the tapestry of stars unfolding above."
]

embeddings = model.encode(sentences)
print(embeddings)

# Excepted output
# tensor([[ 0.3314, -0.2520,  0.4150,  ...,  0.1575, -0.1235, -0.1226],
#         [ 0.5128, -0.0051,  0.2179,  ...,  0.1010,  0.1654, -0.3872]])
 ```

 ## Evaluation

### Run Evaluation with SentEval
```
python eval_senteval.py \
    --model_name_or_path Singhoo/denosent-bert-base \
    --task_set sts \
    --mode test \
```
This checkpoint has slightly higher STS results than those reported in the paper.
```
------ test ------
+-------+-------+-------+-------+-------+--------------+-----------------+-------+
| STS12 | STS13 | STS14 | STS15 | STS16 | STSBenchmark | SICKRelatedness |  Avg. |
+-------+-------+-------+-------+-------+--------------+-----------------+-------+
| 75.48 | 83.82 | 77.54 | 84.76 | 80.16 |    81.20     |      73.97      | 79.56 |
+-------+-------+-------+-------+-------+--------------+-----------------+-------+
```

### Run evaluation with MTEB
```
python eval_mteb.py \
    --model_name_or_path Singhoo/denosent-bert-base \
```
Evaluation results for MTEB will appear in a separate directory `mteb_results`.

## Train Your Own DenoSent Models
Run the following command to train your own models. Try out different hyperparameters as you like. The dataset will be automatically downloaded from Huggingface.
```
python \
    train.py \
    --train_dataset Singhoo/denosent_data \
    --torch_compile True \
    --model_name_or_path bert-base-uncased \
    --max_length 32 \
    --decoder_num_layers 16 \
    --decoder_num_heads 1 \
    --decoder_target_dropout 0.825 \
    --pooler mask \
    --output_dir results \
    --overwrite_output_dir \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 256 \
    --learning_rate 4e-5 \
    --lr_scheduler_type constant_with_warmup \
    --do_train \
    --do_eval \
    --evaluation_strategy steps \
    --eval_steps 50 \
    --save_strategy steps \
    --save_steps 50 \
    --num_train_epochs 1 \
    --metric_for_best_model eval_avg_sts \
    --prompt_format '"[X]" means [MASK].' \
    --do_contrastive \
    --do_generative \
    --save_total_limit 1 \
    --contrastive_temp 0.05 \
    --warmup_steps 500 \
    --contrastive_weight 5 \
    --generative_weight 7 \
    --max_steps 5000 \
    --load_best_model_at_end \
```

## Acknowledgements

We use the [SentEval toolkit](https://github.com/facebookresearch/SentEval) and the [MTEB toolkit](https://github.com/embeddings-benchmark/mteb) for evaluations, and we adopt the modified version of SenteEal from the [SimCSE repository](https://github.com/princeton-nlp/SimCSE).