
from mteb import MTEB
import argparse
from transformers import BertConfig, AutoModel, AutoTokenizer
import torch
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)

class MtebWrapper:
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        self.model = AutoModel.from_pretrained(args.model_name_or_path).to(self.device)
        self.model.eval()

    def encode(self, sentences, batch_size=32, **kwargs):
        """ Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """
        prompt_format = self.args.prompt_format
        all_embeddings = []
        # torch.argsort(torch.tensor([len(sen) for sen in sentences]))
        # length_sorted_idx = torch.argsort(torch.tensor([len(sen) for sen in sentences]))
        length_sorted_idx = np.argsort([len(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in range(0, len(sentences), batch_size):
            sentences_batch = sentences_sorted[start_index:start_index+batch_size]
            # print(self.tokenizer(sentences_batch, padding=True, truncation=True, max_length=500, return_tensors='pt', add_special_tokens=False).input_ids.shape)
            if self.args.pooler == 'mask':
                sentences_batch = self.tokenizer.batch_decode(self.tokenizer(sentences_batch, padding=True, truncation=True, max_length=500, return_tensors='pt', add_special_tokens=False).input_ids, skip_special_tokens=True)
                sentences_batch = [prompt_format.replace('[X]', s).replace('[MASK]', self.tokenizer.mask_token) for s in sentences_batch]
            inputs = self.tokenizer(sentences_batch, padding=True, truncation=True, return_tensors="pt")
            if self.args.pooler == 'mask':
                mask_token_id_idx = (inputs['input_ids']==self.tokenizer.mask_token_id).nonzero(as_tuple=True)[1].unsqueeze(-1).to(self.device)
            inputs = {k: v.to(self.device) for k,v in inputs.items()}
            # Get the embeddings
            with torch.no_grad():
                last_hidden_state = self.model(**inputs, output_hidden_states=True, return_dict=True).last_hidden_state
                if self.args.pooler == 'cls':
                    embeddings = last_hidden_state[:, 0, :]
                elif self.args.pooler == 'mask':
                    embeddings = torch.gather(last_hidden_state, dim=1, index=mask_token_id_idx.unsqueeze(-1).repeat(1, 1, last_hidden_state.size()[-1])).squeeze()
                else:
                    raise NotImplementedError()
            all_embeddings.extend(embeddings.cpu())
        all_embeddings = torch.stack([all_embeddings[idx] for idx in np.argsort(length_sorted_idx)])
        return all_embeddings

TASKS = [
    # 'STS12',
    # 'STS13',
    # 'STS14',
    # 'STS15',
    # 'STS16',
    # 'STSBenchmark',
    # 'SICK-R',
]

TASK_LIST_CLASSIFICATION = [
    "AmazonCounterfactualClassification",
    # "AmazonPolarityClassification",
    "AmazonReviewsClassification",
    "Banking77Classification",
    "EmotionClassification",
    # "ImdbClassification",
    "MassiveIntentClassification",
    "MassiveScenarioClassification",
    "MTOPDomainClassification",
    "MTOPIntentClassification",
    "ToxicConversationsClassification",
    "TweetSentimentExtractionClassification",
]

TASK_LIST_CLUSTERING = [
    "ArxivClusteringP2P",
    "ArxivClusteringS2S",
    "BiorxivClusteringP2P",
    "BiorxivClusteringS2S",
    "MedrxivClusteringP2P",
    "MedrxivClusteringS2S",
    "RedditClustering",
    "RedditClusteringP2P",
    "StackExchangeClustering",
    "StackExchangeClusteringP2P",
    "TwentyNewsgroupsClustering",
]

TASK_LIST_PAIR_CLASSIFICATION = [
    "SprintDuplicateQuestions",
    "TwitterSemEval2015",
    "TwitterURLCorpus",
]

TASK_LIST_RERANKING = [
    "AskUbuntuDupQuestions",
    "MindSmallReranking",
    "SciDocsRR",
    "StackOverflowDupQuestions",
]

TASK_LIST_RETRIEVAL = [
    "ArguAna",
    "ClimateFEVER",
    "CQADupstackAndroidRetrieval",
    "CQADupstackEnglishRetrieval",
    "CQADupstackGamingRetrieval",
    "CQADupstackGisRetrieval",
    "CQADupstackMathematicaRetrieval",
    "CQADupstackPhysicsRetrieval",
    "CQADupstackProgrammersRetrieval",
    "CQADupstackStatsRetrieval",
    "CQADupstackTexRetrieval",
    "CQADupstackUnixRetrieval",
    "CQADupstackWebmastersRetrieval",
    "CQADupstackWordpressRetrieval",
    "DBPedia",
    "FEVER",
    "FiQA2018",
    "HotpotQA",
    "MSMARCO",
    "NFCorpus",
    "NQ",
    "QuoraRetrieval",
    "SCIDOCS",
    "SciFact",
    "Touche2020",
    "TRECCOVID",
]

TASK_LIST_STS = [

    "SICK-R",
    "STS12",
    "STS13",
    "STS14",
    "STS15",
    "STS16",

    "STSBenchmark",

]

TASK_LIST = TASK_LIST_CLASSIFICATION + TASK_LIST_CLUSTERING + TASK_LIST_PAIR_CLASSIFICATION + TASK_LIST_RERANKING + TASK_LIST_RETRIEVAL + TASK_LIST_STS

TASKS = [
    # "BIOSSES",
    "SICK-R",
    "STS12",
    "STS13",
    "STS14",
    "STS15",
    "STS16",
    # "STS17",
    "STSBenchmark",
    "AskUbuntuDupQuestions",
    "MindSmallReranking",
    "SciDocsRR",
    "StackOverflowDupQuestions",
    "QuoraRetrieval",

]
TASKS = TASKS + TASK_LIST_CLASSIFICATION
RTASKS = [
    "AskUbuntuDupQuestions",
    "MindSmallReranking",
    "SciDocsRR",
    "StackOverflowDupQuestions",
    "QuoraRetrieval",
]
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, 
            help="Transformers' model name or path")
    parser.add_argument("--pooler", type=str, 
            choices=['cls', 'mask'], 
            default='cls', )
    parser.add_argument("--prompt_format", type=str,)
    parser.add_argument("--decoder_target_layer", type=int)
    parser.add_argument("--decoder_num_layers", type=int)
    parser.add_argument("--decoder_num_heads", type=int)
    parser.add_argument("--decoder_target_dropout", type=float)
    parser.add_argument("--add_projector", type=bool)
    parser.add_argument("--normalize_embedding", type=bool, default=False)
    parser.add_argument("--decoder_random_target", type=bool, default=False)
    parser.add_argument("--shuffle_target", type=bool, default=False)
    parser.add_argument("--use_raw_embeddings", type=bool, default=False)
    parser.add_argument("--shift_right", type=bool, default=False)

    args = parser.parse_args()
#     model = AutoModel.from_pretrained(args.model_name_or_path)
    # model = MtebWrapper(args)
    config = SelfSentConfig.from_pretrained(args.model_name_or_path)
    model = SelfSentModel.from_pretrained(args.model_name_or_path, config=config)
    model = model.to("cuda")
    model.eval()
    # evaluation = MTEB(task_categories=['STS'])
    # evaluation.run(model, overwrite_results=True, batch_size=64, eval_splits=["test"], output_folder='mteb_results/'+args.model_name_or_path.split('/')[-1])

    # for task in TASKS:
    #     print("Running task: ", task)
    eval_splits = ["test"]
    evaluation = MTEB(tasks=TASK_LIST_STS, task_langs=["en"], task_categories=['S2S'])
    results = evaluation.run(model, overwrite_results=True, batch_size=64, eval_splits=eval_splits, output_folder='mteb_results/'+args.model_name_or_path.split('/')[-1])
    sts12 = results['STS12']['test']['cos_sim']['spearman']
    sts13 = results['STS13']['test']['cos_sim']['spearman']
    sts14 = results['STS14']['test']['cos_sim']['spearman']
    sts15 = results['STS15']['test']['cos_sim']['spearman']
    sts16 = results['STS16']['test']['cos_sim']['spearman']
    sickr = results['SICK-R']['test']['cos_sim']['spearman']
    stsb = results['STSBenchmark']['test']['cos_sim']['spearman']
    avg_sts = (sts12 + sts13 + sts14 + sts15 + sts16 + sickr + stsb) / 7
    print("avg_sts: ", avg_sts)
if __name__ == '__main__':
    main()