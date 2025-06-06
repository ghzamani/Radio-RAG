from evaluation import compute_scores
import re
from utills import read_jsonl_as_dict, extract_sections


def main():
    gold_impressions = list(read_jsonl_as_dict('gold_impressions.jsonl').values())
    gold_findings = list(read_jsonl_as_dict('gold_findings.jsonl').values())
    predicted_reports = read_jsonl_as_dict('predicted_reports.jsonl')
    predicted_impressions = []
    predicted_findings = []
    for report in predicted_reports.values():
        # TODO what if model did not generate findings/impression?
        findings, impression = extract_sections(report)
        predicted_findings.append(findings)
        predicted_impressions.append(impression)

    impression_scores = compute_scores(gold_impressions, predicted_impressions)
    findings_scores = compute_scores(gold_findings, predicted_findings)
    print("impression:", impression_scores)
    print("findings:", findings_scores)


if __name__ == '__main__':
    main()