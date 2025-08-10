from evaluation import compute_scores
from utills import read_jsonl_as_dict, extract_sections

def lower_case(my_list):
    return [item.lower().strip() for item in my_list]

def evaluate_reports(dir_name: str):
    gold_impressions = list(read_jsonl_as_dict(dir_name + 'gold_impressions.jsonl').values())
    gold_findings = list(read_jsonl_as_dict(dir_name + 'gold_findings.jsonl').values())
    predicted_reports = read_jsonl_as_dict(dir_name + 'predicted_reports.jsonl')
    predicted_impressions = []
    predicted_findings = []
    for report in predicted_reports.values():
        # TODO what if model did not generate findings/impression? use structured outputs?
        findings, impression = extract_sections(report)
        predicted_findings.append(findings)
        predicted_impressions.append(impression)

    # convert ground truth & predictions to lower
    gold_impressions = lower_case(gold_impressions)
    gold_findings = lower_case(gold_findings)
    predicted_impressions = lower_case(predicted_impressions)
    predicted_findings = lower_case(predicted_findings)

    print("Computing impression scores")
    impression_scores = compute_scores(gold_impressions, predicted_impressions,
                                       output_name=dir_name + "impression_scores.json")
    print("Computing findings scores")
    findings_scores = compute_scores(gold_findings, predicted_findings,
                                     output_name=dir_name + "findings_scores.json")
    print("impression:", impression_scores)
    print("findings:", findings_scores)

    return impression_scores, findings_scores


if __name__ == '__main__':
    evaluate_reports("study_exp6/")