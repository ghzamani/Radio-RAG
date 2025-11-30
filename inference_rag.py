import os
# os.environ['HF_HOME'] = "/home/m_nobakhtian/mmed/hf_cache"

import faiss
import pickle
import torch, torchvision
import numpy as np
import torchxrayvision as xrv
from openai import OpenAI
from datasets import load_dataset
from tqdm import tqdm
import json
import random
import yaml
import wandb
from collections import Counter

# from utills import load_test_bench, xray_transform, load_radio_bench, extract_sections
from utills import xray_transform
from eval import evaluate_reports

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

RETRIEVAL_SYS_PROMPT = """You are an expert radiologist specializing in chest X-rays.  
Your task is to generate only the FINDINGS and IMPRESSION sections of a chest X-ray report.  

Follow these rules:
- Base your report on the provided FINDINGS and retrieved similar reports for guidance.  
- Write in a concise, professional tone used in real radiology reports.
- Do NOT copy text verbatim from retrieved reports unless medically appropriate.  
- Do NOT mention any numeric scores, probabilities, confidence values, or thresholds in the report text.
- Do NOT include patient identifiers, clinical history, or section headers other than FINDINGS and IMPRESSION.  
- Use complete sentences and standard medical terminology.  
"""

RETRIEVAL_WITH_NEGATIVE_SYS_PROMPT = """You are an expert radiologist specializing in chest X-rays.  
Your task is to generate only the FINDINGS and IMPRESSION sections of a chest X-ray report.  

Follow these rules:
- Base your report on the **provided FINDINGS** and **retrieved similar reports** for guidance.  
- Retrieved **dissimilar** reports are provided as **explicit negative examples**. These describe cases that are different from the current patient.  
    -- DO NOT copy, paraphrase, or include content from dissimilar reports unless it is also supported by the predicted findings or similar reports.  
    -- Treat them only as guidance for what to AVOID in this case.  
- Always prefer evidence from predicted labels and similar reports over anything else.  
- Write in a concise, professional tone used in real radiology reports.
- Do NOT copy text verbatim from retrieved reports unless medically appropriate.  
- Do NOT mention any numeric scores, probabilities, confidence values, or thresholds in the report text.
- Do NOT include patient identifiers, clinical history, or section headers other than FINDINGS and IMPRESSION.  
- Use complete sentences and standard medical terminology.  
"""

# only used for (partial and related) mode
PARTIAL_RELATED_SYS_PROMPT = """You are an expert radiologist specializing in chest X-rays.  
You will be given the top detected symptoms for the patient.  
For some symptoms, you will also receive a set of retrieved reports that are most relevant to that specific symptom.  
These retrieved reports are grouped by symptom so you can clearly see which ones relate to which finding.  

Your task is to generate only the FINDINGS and IMPRESSION sections of a chest X-ray report.  

Follow these rules:
- Use the provided FINDINGS and retrieved symptom-specific reports as guidance.  
- Consider the relevance of each retrieved report to its symptom, but integrate the information into a single coherent report.  
- Write in a concise, professional tone used in real radiology reports.  
- Do NOT copy text verbatim from retrieved reports unless medically appropriate.  
- Do NOT mention any numeric scores, probabilities, confidence values, or thresholds in the report text.  
- Do NOT include patient identifiers, clinical history, or section headers other than FINDINGS and IMPRESSION.  
- Use complete sentences and standard medical terminology.  
"""

NON_RETRIEVAL_SYS_PROMPT = """You are an expert radiologist specializing in chest X-rays.  
Your task is to generate only the FINDINGS and IMPRESSION sections of a chest X-ray report.  

Follow these rules:
- Base your report only on the provided FINDINGS.
- Write in a concise, professional tone used in real radiology reports.
- Do NOT mention any numeric scores, probabilities, confidence values, or thresholds in the report text.
- Do NOT include patient identifiers, clinical history, or section headers other than FINDINGS and IMPRESSION.
- Use complete sentences and standard medical terminology. 
"""


def jaccard_batch(query, matrix):
    """
    Compute Jaccard similarity between a binary query and a binary matrix.
    query: shape (d,), binary 0/1
    matrix: shape (n, d), binary 0/1
    returns: shape (n,)
    """
    intersection = np.sum(np.logical_and(matrix, query), axis=1)
    union = np.sum(np.logical_or(matrix, query), axis=1)
    # avoid division by zero
    return intersection / (union + 1e-10)

def hamming_similarity(query, matrix):
    # similarity = 1 - normalized distance
    distance = np.sum(matrix != query, axis=1) / matrix.shape[1]
    return 1 - distance

def cosine_similarity(query, matrix):
    """
    Compute cosine similarity between query and matrix.
    query: shape (d,)
    matrix: shape (n, d)
    returns: shape (n,)
    """
    dot = matrix @ query
    query_norm = np.linalg.norm(query)
    matrix_norms = np.linalg.norm(matrix, axis=1)
    return dot / (matrix_norms * query_norm + 1e-10)

def retrieve_with_backup(query_bin, query_float, db_bin, db_float, reports,
                         k=5, primary="jaccard"):
    # pick primary similarity
    if primary == "jaccard":
        sims = jaccard_batch(query_bin, db_bin)
    elif primary == "hamming":
        sims = hamming_similarity(query_bin, db_bin)
    else:
        raise ValueError("primary metric must be 'jaccard' or 'hamming'")

    cosine_sims = cosine_similarity(query_float, db_float)
    # argsort by Jaccard/Hamming first (descending), then cosine (descending)
    indices = np.lexsort((-cosine_sims, -sims))
    topk = indices[:k]
    return [reports[i] for i in topk]


def retrieve_most_similar(predicted_label_vector, vector_index, reports,
                          k=5, metric="l2"):
    query_vector = np.array(predicted_label_vector).astype('float32').reshape(1, -1)
    # only used for cosine similarity
    if metric == 'cosine':
        norm = query_vector / np.linalg.norm(query_vector, axis=1, keepdims=True)
        query_vector = query_vector / np.maximum(norm, 1e-10)

    # Search
    _, indices = vector_index.search(query_vector, k=k)  # get top 5 similar

    # Retrieve reports
    retrieved_reports = [reports[idx] for idx in indices[0]]
    return retrieved_reports


def retrieve_for_top_symptoms(query_vector, symptom_database_vectors, reports, label_names,
                              top_k_symptoms=3, retrieved_k_reports=3, randomly=False):
    """
    Considers top-k symptoms, and retrieves k reports for each
    query_vector: (1, d=11) numpy array with float values
    symptom_database_vectors: (n, d) numpy array used to build the index,
    reports: (n) original reports,
    top_k_symptoms: number of symptoms which are important
    retrieved_k_reports: number of reports to be retrieved per symptom

    Returns:
        Dictionary mapping each top symptom to a list of similar reports
    """

    similar_reports_per_symptom = {}

    # find top-k symptoms
    most_bold_indices = np.argsort(query_vector)[::-1][:top_k_symptoms]
    if not randomly:
        # iterate through each and find the most similar reports
        for i in most_bold_indices:
            distances = np.abs(symptom_database_vectors[:, i] - query_vector[i])
            similar_reports_indices = np.argsort(distances)[:retrieved_k_reports]
            similar_reports = [reports[idx] for idx in similar_reports_indices]
            symptom_name = label_names[i]
            similar_reports_per_symptom[symptom_name] = similar_reports

    else:
        # select reports randomly
        for s, i in enumerate(most_bold_indices):
            # setting seed leads to same reports for all samples
            # random.seed(s)
            similar_reports = random.choices(reports, k=retrieved_k_reports)
            symptom_name = label_names[i]
            similar_reports_per_symptom[symptom_name] = similar_reports

    return similar_reports_per_symptom


def format_labels(pairs, include_scores):
    """Format (value, name) pairs with or without scores."""
    return [
        f"- {n}: {round(v, 2)}" if include_scores else f"- {n}"
        for v, n in pairs
    ]

def format_grouped_reports(dict_symptom_reports, prompt_type='non_related'):
    """
    dict_symptom_reports: { symptom_name: [report1, report2, ...], ... }
    returns formatted multiline string
    """
    similar_reports = ""
    if prompt_type == 'non_related':
        # Add retrieved reports without any special structure
        non_structured_reports = []
        for reports in dict_symptom_reports.values():
            non_structured_reports.extend(reports)
        # todo sort these reports in a way cause llm cares about placement
        for i, report in enumerate(non_structured_reports):
            similar_reports += f"\n--- Report {i + 1} ---\n"
            similar_reports += f"{report}\n"

    elif prompt_type == 'related':
        parts = []
        for symptom, reports in dict_symptom_reports.items():
            parts.append(f"===== Symptom: {symptom} =====")
            for i, rep in enumerate(reports):
                parts.append(f"--- Report {i + 1} ---\n{rep}\n")
        similar_reports = "\n".join(parts)
    else:
        raise ValueError('prompt_type must be "non_related" or "related"')

    return similar_reports


def build_prompt(label_vector, label_names, dict_symptom_reports,
                 prompt_type='non_related',
                 threshold=0.55,
                 retrieved=True,
                 include_label_scores=True,
                 need_sort=True,
                 negative_dict_symptom_reports=None):
    # Convert label vector to readable findings
    label_list = label_vector.tolist()
    # Filter out None names and create valid pairs
    valid_pairs = [(v, n) for v, n in zip(label_list, label_names)]
    # Sort descending
    if need_sort:
        valid_pairs.sort(reverse=True, key=lambda pair: pair[0])

    # Filter above threshold
    above_threshold = [(v, n) for v, n in valid_pairs if v >= threshold]

    if above_threshold:
        predicted_findings = "\n".join(format_labels(above_threshold, include_label_scores))
    else:
        # TODO what if all labels where under threshold? predicted_findings would be empty
        # Fallback to top 3 by score
        top3 = sorted(valid_pairs, key=lambda x: x[0], reverse=True)[:3]
        predicted_findings = "\n".join(format_labels(top3, include_label_scores))

    prompt = f"Findings:\n{predicted_findings}\n"

    if retrieved:
        similar_reports = format_grouped_reports(dict_symptom_reports, prompt_type)
        prompt += f"\nRetrieved similar reports:\n{similar_reports}"

    if negative_dict_symptom_reports:
        dissimilar_reports = format_grouped_reports(negative_dict_symptom_reports, prompt_type)
        prompt += f"\nRetrieved dissimilar reports: (negative references)\n{dissimilar_reports}"

    prompt += "\nGenerate a new FINDINGS and IMPRESSION section for this case."

    return prompt


def call_gpt(client, user_prompt, system_prompt, model="gpt-4o-mini"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0,
        max_tokens=300
    )
    return response.choices[0].message.content


def save_result(input_list, file_path):
    # create directory if it doesn't exist
    dir_name = os.path.dirname(file_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    with open(file_path, "w") as f:
        for idx, item in enumerate(input_list):
            data = {idx: item}
            f.write(json.dumps(data) + '\n')


def load_model_and_resources(base_path='', index_type='l1'):
    transform = torchvision.transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224)
    ])
    model = xrv.models.DenseNet(weights="densenet121-res224-mimic_ch")
    model = model.to(DEVICE)

    try:
        index = faiss.read_index(f"{base_path}/{index_type}.index")
        print(f"index {index_type} loaded.")
    except:
        print("Index type must be l1, l2, or cosine")
        # for partial, jaccard & hamming we wouldn't use index
        index = None

    with open(f"{base_path}/index_to_report.pkl", "rb") as f:
        original_reports = pickle.load(f)
    db_vectors = np.load(f"{base_path}/symptoms_vectors.npy")
    non_empty_indices = [i for i, name in enumerate(model.pathologies) if name]

    return model, transform, non_empty_indices, index, original_reports, db_vectors


def predict_retrieve(model, transform, samples, db_vectors, original_reports,
                     top_k_symptoms=3, retrieved_k_reports=3, randomly=False):
    # find indices of pathologies (11 out of 18)
    non_empty_indices = [i for i, name in enumerate(model.pathologies) if name]
    # a list in which each item is a list of retrieved reports
    samples_similar_reports = []
    predicted_labels = []
    # if we want to read a file instead of predicting,
    # then need a mapping from study_id to impression and findings
    for sample in tqdm(samples,
                       desc="Predicting pathologies on test data"):
        # predict labels for input image
        transformed = xray_transform(sample['image'], transform).to(DEVICE)
        # predict on dataset
        pred = model(transformed).flatten()
        pred = pred.cpu().detach().numpy()
        # Filter the vector using these indices
        pred = pred[non_empty_indices]
        predicted_labels.append(pred)
        # similar_reports = retrieve_most_similar(pred, index, original_reports, k=5)

        # TODO pay attention to randomly boolean
        dict_symptom_reports = retrieve_for_top_symptoms(pred,
                                                         db_vectors,
                                                         original_reports,
                                                         # need to add label_names here
                                                         top_k_symptoms=top_k_symptoms,
                                                         retrieved_k_reports=retrieved_k_reports,
                                                         randomly=randomly)
        similar_reports = []
        for reps in dict_symptom_reports.values():
            similar_reports.extend(reps)
        samples_similar_reports.append(similar_reports)
    return predicted_labels, samples_similar_reports

def run_gpt_reporting_step(
        config,
        predicted_labels,
        samples_symptom_reports,
        label_names,
        threshold=0.65,
        retrieve_needed=True,
        include_label_scores=False):
    """
    Calls GPT to generate reports from predicted labels and similar reports.
    Saves the gold and predicted reports to disk.

    Parameters:
    - predicted_labels: list of label vectors
    - samples_similar_reports: list of retrieved text samples per case
    - dir_name: where to save the outputs (should end with '/')
    - retrieve_needed: whether to tell the prompt builder that retrieval was used
    """
    client = OpenAI(
        api_key=config['openai_api_key']
    )

    prompt_type = config.get('prompt_type', 'non_related')
    system_prompt = RETRIEVAL_SYS_PROMPT
    if not config['use_retrieval']:
        system_prompt = NON_RETRIEVAL_SYS_PROMPT
    if config['retrieval_mode'] == 'partial' and prompt_type == 'related':
        system_prompt = PARTIAL_RELATED_SYS_PROMPT
    print("system prompt:\n", system_prompt)

    predicted_reports = []
    prompts = []

    for idx, (labels, similar_reports) in enumerate(tqdm(zip(predicted_labels, samples_symptom_reports),
                                                         desc="Requesting GPT")):
        prompt = build_prompt(labels, label_names, similar_reports,
                              prompt_type=prompt_type,
                              threshold=threshold,
                              retrieved=retrieve_needed,
                              include_label_scores=include_label_scores)
        new_report = call_gpt(client, prompt, system_prompt)

        prompts.append(prompt)
        predicted_reports.append(new_report)

        # save results right after prediction
        with open(config['output_path'] + "predicted_reports.jsonl", "a") as pred_f:
            pred_f.write(json.dumps({idx: new_report}) + "\n")

        with open(config['output_path'] + "prompts.jsonl", "a") as prompt_f:
            prompt_f.write(json.dumps({idx: prompt}) + "\n")
        # break  # for debugging

def run_gpt_reporting_step_negative(
        config,
        predicted_labels,
        samples_symptom_reports,
        negative_samples_symptom_reports,
        label_names,
        threshold=0.65,
        retrieve_needed=True,
        include_label_scores=False):

    client = OpenAI(
        api_key=config['openai_api_key']
    )

    prompt_type = config.get('prompt_type', 'non_related')
    # system_prompt = RETRIEVAL_SYS_PROMPT
    # if not config['use_retrieval']:
    #     system_prompt = NON_RETRIEVAL_SYS_PROMPT
    # if config['retrieval_mode'] == 'partial' and prompt_type == 'related':
    #     system_prompt = PARTIAL_RELATED_SYS_PROMPT
    system_prompt = RETRIEVAL_WITH_NEGATIVE_SYS_PROMPT
    print("system prompt:\n", system_prompt)

    predicted_reports = []
    prompts = []

    for idx, (labels, similar_reports, dissimilar_reports) in enumerate(tqdm(zip(predicted_labels,
                                                                                 samples_symptom_reports,
                                                                                 negative_samples_symptom_reports),
                                                                             desc="Requesting GPT")):
        prompt = build_prompt(labels, label_names, similar_reports,
                              prompt_type=prompt_type,
                              threshold=threshold,
                              retrieved=retrieve_needed,
                              include_label_scores=include_label_scores,
                              negative_dict_symptom_reports=dissimilar_reports)
        new_report = call_gpt(client, prompt, system_prompt)

        prompts.append(prompt)
        predicted_reports.append(new_report)

        # save results right after prediction
        with open(config['output_path'] + "predicted_reports.jsonl", "a") as pred_f:
            pred_f.write(json.dumps({idx: new_report}) + "\n")

        with open(config['output_path'] + "prompts.jsonl", "a") as prompt_f:
            prompt_f.write(json.dumps({idx: prompt}) + "\n")
        # break  # for debugging


def count_non_zero(predictions, threshold=0.65):
    """count number of 1s of each study after converting to binarized version, just for analysis"""
    predictions_np = np.array(predictions)
    binary_predictions_np = (predictions_np >= threshold).astype(int)
    counts_per_row = np.count_nonzero(binary_predictions_np == 1, axis=1)
    count_distribution = Counter(counts_per_row)
    print("1_count, total")
    for key, c in count_distribution.items():
        print(key, c)
    return count_distribution

def run_experiment(config, aggregated_scores, study_ground_truth):
    """
    I created a dict mapping study_id -> gold_impression, gold_findings in colab
    for those samples which have both impression and findings
    also, we have a dict mapping study_id -> txr vector (max values)

    - read both dicts
    - iterate through first one, find it in the second
    - send request to gpt
    """

    retrieval_mode = config['retrieval_mode']
    use_retrieval = config['use_retrieval']
    randomly = config['randomly']
    index_base_path = config['index_base_path']
    retrieved_k_reports = config.get('retrieved_k_reports', None)
    # only for partial
    top_k_symptoms = config.get('top_k_symptoms', None)
    # only for whole
    binarized_retrieval = config.get('binarized_retrieval', None)
    similarity_metric = config.get('similarity_metric', None)
    negative_retrieval = config.get('negative_retrieval', False)
    # used for prompt so needed for both modes
    include_label_scores = config['include_label_scores']
    threshold = config['threshold']

    # --- Load model + DB ---
    # need to load specific vector database
    model, _, _, index, original_reports, db_vectors = load_model_and_resources(
        base_path=index_base_path, index_type=similarity_metric
    )
    label_names = [n for n in model.pathologies if n.strip()]

    gold_impression = []
    gold_findings = []
    predicted_labels = []
    # each item is a dict for one sample, with symptom as key and retrieved reports as value
    samples_symptom_reports = []
    samples_symptom_reports_negative = []
    # [
    #   {
    #     "Atelectasis": ["Report 1 for Atelectasis", "Report 2 for Atelectasis"],
    #     "Effusion": ["Report 1 for Effusion", "rep 2"]
    #   }, ...
    # ]

    count = 0
    for study_id, gt in tqdm(study_ground_truth.items(), desc="Iterating through studies"):
        gold_impression.append(gt["impression"])
        gold_findings.append(gt["findings"])
        # get txr vector
        txr_predicted = aggregated_scores[study_id]
        predicted_labels.append(txr_predicted)

        if use_retrieval:
            if retrieval_mode not in {"whole", "partial"}:
                raise ValueError("retrieval_mode must be 'whole' or 'partial'")

            dict_symptom_reports = {}
            if retrieval_mode == "whole":
                if (similarity_metric is None
                        or binarized_retrieval is None
                        or retrieved_k_reports is None):
                    raise Exception("similarity_metric or binarized_retrieval or retrieved_k_reports not defined")
                # --- Apply binarization for retrieval step ---
                if binarized_retrieval:
                    txr_for_retrieval = (txr_predicted >= threshold).astype(int)

                    # binarize train vectors
                    db_vectors_binarized = (db_vectors >= threshold).astype(int)
                    similar_reports = retrieve_with_backup(txr_for_retrieval, txr_predicted,
                                                           db_vectors_binarized, db_vectors,
                                                           original_reports,
                                                           primary=similarity_metric, k=retrieved_k_reports)
                    # negative context
                    if negative_retrieval:
                        negative_txr = 1 - txr_for_retrieval
                        dissimilar_reports = retrieve_with_backup(negative_txr, txr_predicted,
                                                                  db_vectors_binarized, db_vectors,
                                                                  original_reports,
                                                                  primary=similarity_metric, k=retrieved_k_reports)
                        dict_symptom_reports_negative = {'dummy': dissimilar_reports}
                        samples_symptom_reports_negative.append(dict_symptom_reports_negative)

                else:
                    similar_reports = retrieve_most_similar(txr_predicted, index, original_reports,
                                                            k=retrieved_k_reports)
                # convert to dict format for rest of the code
                dict_symptom_reports = {'dummy': similar_reports}

            elif retrieval_mode == "partial":
                if (top_k_symptoms is None
                        or retrieved_k_reports is None):
                    raise Exception("top_k_symptoms or retrieved_k_reports not defined")
                dict_symptom_reports = retrieve_for_top_symptoms(
                    txr_predicted,
                    db_vectors,
                    original_reports,
                    label_names=label_names,
                    top_k_symptoms=top_k_symptoms,
                    retrieved_k_reports=retrieved_k_reports,
                    randomly=randomly
                )
            # is there any possibility that dict_symptom_reports remains empty?
            if not dict_symptom_reports:
                raise Exception("dict_symptom_reports not defined")
            samples_symptom_reports.append(dict_symptom_reports)

        # create a dummy similar report for run_gpt_reporting_step function
        else:
            samples_symptom_reports = [{} for _ in range(len(predicted_labels))]

        count += 1
        # if count >= 5:
        #     break
    if retrieval_mode == "whole" and binarized_retrieval:
        count_non_zero(predicted_labels, threshold)

    # save ground-truth impression & findings into file
    save_result(gold_impression, config['output_path'] + "gold_impressions.jsonl")
    save_result(gold_findings, config['output_path'] + "gold_findings.jsonl")

    if negative_retrieval:
        run_gpt_reporting_step_negative(
            config,
            predicted_labels,
            samples_symptom_reports,
            samples_symptom_reports_negative,
            label_names,
            threshold=threshold,
            retrieve_needed=use_retrieval,
            include_label_scores=include_label_scores
        )
    else:
        # --- GPT reporting step ---
        run_gpt_reporting_step(
            config,
            predicted_labels,
            samples_symptom_reports,
            label_names,
            threshold=threshold,
            retrieve_needed=use_retrieval,
            include_label_scores=include_label_scores
        )

def load_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def resume_wandb(path, run_id):
    # be careful about the run id!
    config = load_config(path)
    project_name = config['wandb_project']
    os.environ["WANDB_API_KEY"] = config["wandb_api_key"]
    wandb.init(project=project_name, id=run_id, resume="must")

    impression_scores, findings_scores = evaluate_reports(config['output_path'])
    wandb.log({
        "impression": impression_scores,
        "findings": findings_scores,
    })
    wandb.finish()


# def main():
def exp_with_eval(conf_path):
    # path = "configs/whole_l2_no_score.yaml"
    # path = "configs/whole_negative_jaccard_include_score.yaml"
    # config = load_config(path)
    config = load_config(conf_path)

    split = "test"
    # I decided to use mean, it is also possible to use max pool
    with open(f"/root/data/mean_scores_{split}.pkl", "rb") as f:
        # 3269 for test
        aggregated_scores = pickle.load(f)
    # a dict mapping study_id to ground_truth impression & findings
    # only for samples having both
    with open(f"/root/data/study_to_gold_{split}.pkl", "rb") as f:
        # 1624 for test
        study_ground_truth = pickle.load(f)

    # swin_chexpert(study_ground_truth)
    # initialize wandb
    os.environ["WANDB_API_KEY"] = config["wandb_api_key"]
    wandb.init(project=config['wandb_project'],
               name=config['wandb_run_name'],
               config=config)

    run_experiment(config, aggregated_scores, study_ground_truth)
    impression_scores, findings_scores = evaluate_reports(config['output_path'])

    wandb.log({
        "impression": impression_scores,
        "findings": findings_scores,
    })
    wandb.finish()


def swin_chexpert(study_ground_truth):
    with open('/root/codes/radio/data/study_impression_swin_test.pkl', 'rb') as f:
        scores = pickle.load(f)
    aggregated_scores = {}
    for std_id, views in scores.items():
        preds = list(views.values())
        rnd = random.choice(preds)
        aggregated_scores[std_id] = rnd

    gold_impression = []
    gold_findings = []
    preds = []
    for study_id, gt in tqdm(study_ground_truth.items(), desc="Iterating through studies"):
        gold_impression.append(gt["impression"])
        # gold_findings.append(gt["findings"])

        pred = aggregated_scores[study_id]
        preds.append(pred)

    base = "./results/chexpert/"
    save_result(gold_impression, base + "gold_impressions.jsonl")
    save_result(preds, base + "pred_impression.jsonl")


if __name__ == '__main__':
    exp_with_eval('configs/whole_l2_include_score_less_context.yaml')
    exp_with_eval('configs/whole_l2_include_score_more_context.yaml')
    # for i in range(2, 6):
    #     conf_path = f"configs/baseline_{i}.yaml"
    #     print(conf_path)
    #     exp_with_eval(conf_path)

    # main()
    # resume_wandb("configs/partial_include_score_related.yaml", run_id="50vo48ia")
    # resume_wandb("configs/partial_no_score_simple.yaml", run_id="3q3ghoh4")
    # resume_wandb("configs/whole_l2_include_score.yaml", run_id="2l94ju4p")
    # resume_wandb("configs/whole_cos_include_score.yaml", run_id="zw5dp6tn")
    # resume_wandb("configs/whole_hamming_include_score.yaml", run_id="630zmdut")
    # resume_wandb("configs/whole_hamming_no_score.yaml", run_id="dw14or65")
