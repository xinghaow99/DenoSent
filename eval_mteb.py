
from mteb import MTEB
import argparse
import logging
from model import DenoSentModel
from config import DenoSentConfig

logging.basicConfig(level=logging.INFO)

TASK_CLASSIFICATION = [
    "AmazonCounterfactualClassification",
    "AmazonReviewsClassification",
    "Banking77Classification",
    "EmotionClassification",
    "MassiveIntentClassification",
    "MassiveScenarioClassification",
    "MTOPDomainClassification",
    "MTOPIntentClassification",
    "ToxicConversationsClassification",
    "TweetSentimentExtractionClassification",
]



TASK_RERANKING = [
    "AskUbuntuDupQuestions",
    "MindSmallReranking",
    "SciDocsRR",
    "StackOverflowDupQuestions",
]

TASK_RETRIEVAL = [
    "QuoraRetrieval",
]

TASK_STS = [
    "SICK-R",
    "STS12",
    "STS13",
    "STS14",
    "STS15",
    "STS16",
    "STSBenchmark",
]

TASK_LIST = TASK_CLASSIFICATION + TASK_RERANKING + TASK_RETRIEVAL + TASK_STS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, 
            help="Transformers' model name or path")

    args = parser.parse_args()

    config = DenoSentConfig.from_pretrained(args.model_name_or_path)
    model = DenoSentModel.from_pretrained(args.model_name_or_path, config=config)
    model = model.to("cuda")
    model.eval()

    eval_splits = ["test"]
    evaluation = MTEB(tasks=TASK_LIST, task_langs=["en"], task_categories=['S2S'])
    evaluation.run(model, overwrite_results=True, batch_size=64, eval_splits=eval_splits, output_folder='mteb_results/'+args.model_name_or_path.split('/')[-1])

if __name__ == '__main__':
    main()