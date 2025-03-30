import evaluate
from f1chexbert import F1CheXbert
from radgraph import F1RadGraph
import json

from .bert_scorer import BertScore
from .bleu_scorer import Bleu
from .rouge_scorer import Rouge


def convert_to_dict(refs, hyps):
    refs_dict = {i:[refs[i]] for i in range(len(refs))}
    hyps_dict = {i:[hyps[i]] for i in range(len(hyps))}
    return refs_dict, hyps_dict


def compute_scores(refs, hyps,
                   metrics=["rouge", "bleu", "meteor", "bertscore", "chexbert", "radgraph"]):
    scores = {}
    # If metric is None or empty list
    if metrics is None or not metrics:
        return scores

    assert refs is not None and hyps is not None, \
        "You specified metrics but your evaluation does not return hyps nor refs"

    assert len(refs) == len(hyps), 'refs and hyps must have same length : {} vs {}'.format(len(refs), len(hyps))

    refs_dict, hyps_dict = convert_to_dict(refs, hyps)

    for metric in metrics:
        match metric:
            case "rouge":
                print("Computing Rouge score")
                rouge = Rouge()
                avg_rouge, rouge_scores = rouge.compute_score(refs_dict, hyps_dict)
                scores["Average_Rouge"] = avg_rouge.item()
            case "bleu":
                print("Computing Bleu score")
                bleu = Bleu()
                avg_bleu, bleu_scores = bleu.compute_score(refs_dict, hyps_dict)
                scores["Average_Bleu"] = avg_bleu
            case "meteor":
                print("Computing Meteor score")
                meteor = evaluate.load('meteor')
                meteor_scores = meteor.compute(predictions=hyps, references=refs)['meteor'].item()
                scores["Average_Meteor"] = meteor_scores
            case "bertscore":
                print("Computing Bert score")
                bert_score = BertScore()
                avg_bert_score, bert_scores = bert_score(refs, hyps)
                scores["Average_BertScore"] = avg_bert_score
            case "chexbert":
                print("Computing F1-chexbert score")
                f1chexbert = F1CheXbert()
                accuracy, accuracy_per_sample, chexbert_all, chexbert_5 = f1chexbert(
                    hyps=hyps,
                    refs=refs
                )
                scores["chexbert-5_micro avg_f1-score"] = chexbert_5["micro avg"]["f1-score"]
                scores["chexbert-all_micro avg_f1-score"] = chexbert_all["micro avg"]["f1-score"]
                scores["chexbert-5_macro avg_f1-score"] = chexbert_5["macro avg"]["f1-score"]
                scores["chexbert-all_macro avg_f1-score"] = chexbert_all["macro avg"]["f1-score"]
            case "radgraph":
                print("Computing F1-radgraph score")
                f1radgraph = F1RadGraph(reward_level="all", model_type="radgraph-xl")
                radgraph_simple, radgraph_partial, radgraph_complete = f1radgraph(refs=refs, hyps=hyps)[0]
                scores["radgraph_simple"] = radgraph_simple.item()
                scores["radgraph_partial"] = radgraph_partial.item()
                scores["radgraph_complete"] = radgraph_complete.item()
            case _:
                print(f"{metric} not implemented")

    # with open("scores.json", "w") as f:
    #     json.dump(scores, f, indent=4, sort_keys=True)

    return scores
