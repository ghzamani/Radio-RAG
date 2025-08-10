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

NON_RETRIEVAL_SYS_PROMPT = """You are an expert radiologist specializing in chest X-rays.  
Your task is to generate only the FINDINGS and IMPRESSION sections of a chest X-ray report.  

Follow these rules:
- Base your report only on the provided FINDINGS.
- Write in a concise, professional tone used in real radiology reports.
- Do NOT mention any numeric scores, probabilities, confidence values, or thresholds in the report text.
- Do NOT include patient identifiers, clinical history, or section headers other than FINDINGS and IMPRESSION.
- Use complete sentences and standard medical terminology. 
"""

def retrieve_most_similar(predicted_label_vector, vector_index, reports, k=5):
    query_vector = np.array(predicted_label_vector).astype('float32')
    # only used for cosine similarity
    # query_vector = query_vector / np.linalg.norm(query_vector)

    # Search
    _, indices = vector_index.search(query_vector.reshape(1, -1), k=k)  # get top 5 similar

    # Retrieve reports
    retrieved_reports = [reports[idx] for idx in indices[0]]
    return retrieved_reports


# we need a function that considers top-k symptoms, and retrieves k reports for each
def retrieve_for_top_symptoms(query_vector, symptom_database_vectors, reports,
                              top_k_symptoms=3, retrieved_k_reports=3, randomly=False):
    """
    query_vector: (1, d) numpy array with float values
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
            similar_reports_per_symptom[f"symptom_{i}"] = similar_reports

    else:
        # select reports randomly
        for s, i in enumerate(most_bold_indices):
            # setting seed leads to same reports for all samples
            # random.seed(s)
            similar_reports = random.choices(reports, k=retrieved_k_reports)
            similar_reports_per_symptom[f"symptom_{i}"] = similar_reports

    return similar_reports_per_symptom


def format_labels(pairs, include_scores):
    """Format (value, name) pairs with or without scores."""
    return [
        f"- {n}: {round(v, 2)}" if include_scores else f"- {n}"
        for v, n in pairs
    ]

def build_prompt(label_vector, label_names, retrieved_reports,
                 threshold=0.55,
                 retrieved=True,
                 include_label_scores=True,
                 need_sort=True):
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
        similar_reports = ""
        # Add retrieved reports
        for i, report in enumerate(retrieved_reports):
            similar_reports += f"\n--- Report {i + 1} ---\n"
            similar_reports += f"{report}\n"

        prompt += f"Retrieved similar reports:\n{similar_reports}"

    prompt += "\nGenerate a new FINDINGS and IMPRESSION section for this case."

    return prompt


def call_gpt(client, prompt, retrieved=True, model="gpt-4o-mini"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": RETRIEVAL_SYS_PROMPT if retrieved else NON_RETRIEVAL_SYS_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        max_tokens=300
    )
    return response.choices[0].message.content


def save_result(input_list, file_path):
    # create directory if doesn't exist
    dir_name = os.path.dirname(file_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    with open(file_path, "w") as f:
        for idx, item in enumerate(input_list):
            data = {idx: item}
            f.write(json.dumps(data) + '\n')


def load_model_and_resources(base_path=''):
    transform = torchvision.transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224)
    ])
    model = xrv.models.DenseNet(weights="densenet121-res224-mimic_ch")
    model = model.to(DEVICE)

    index = faiss.read_index(f"{base_path}/l1.index")
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
        samples_similar_reports,
        model,
        gold_impression,
        gold_findings,
        threshold=0.65,
        retrieve_needed=True,
        include_label_scores=False):
    """
    Calls GPT to generate reports from predicted labels and similar reports.
    Saves the gold and predicted reports to disk.

    Parameters:
    - predicted_labels: list of label vectors
    - samples_similar_reports: list of retrieved text samples per case
    - model: the XRV model with .pathologies attribute
    - gold_impression: list of ground truth impressions
    - gold_findings: list of ground truth findings
    - dir_name: where to save the outputs (should end with '/')
    - retrieve_needed: whether to tell the prompt builder that retrieval was used
    """
    client = OpenAI(
        api_key=config['openai_api_key']
    )
    predicted_reports = []
    prompts = []
    label_names = [n for n in model.pathologies if n.strip()]
    for labels, similar_reports in tqdm(zip(predicted_labels, samples_similar_reports),
                                        desc="Requesting GPT"):
        prompt = build_prompt(labels, label_names, similar_reports,
                              threshold=threshold,
                              retrieved=retrieve_needed,
                              include_label_scores=include_label_scores)
        prompts.append(prompt)
        new_report = call_gpt(client, prompt, retrieved=retrieve_needed)
        predicted_reports.append(new_report)
        # break  # for debugging

    save_result(gold_impression, config['output_path'] + "gold_impressions.jsonl")
    save_result(gold_findings, config['output_path'] + "gold_findings.jsonl")
    save_result(predicted_reports, config['output_path'] + "predicted_reports.jsonl")
    save_result(prompts, config['output_path'] + "prompts.jsonl")


def run_experiment(config, aggregated_scores, study_ground_truth):
    """
    I created a dict mapping study_id -> gold_impression, gold_findings in colab
    for those samples which have both impression and findings
    also, we have a dict mapping study_id -> txr vector (max values)

    - read both dicts
    - iterate through first one, find it in the second
    - send request to gpt
    """
    # initialize wandb
    os.environ["WANDB_API_KEY"] = config["wandb_api_key"]
    wandb.init(project=config['wandb_project'],
               name=config['wandb_run_name'],
               config=config)

    # Use config params
    top_k_symptoms = config['top_k_symptoms']
    retrieved_k_reports = config['retrieved_k_reports']
    binarized_retrieval = config['binarized_retrieval']
    include_label_scores = config['include_label_scores']
    retrieval_mode = config['retrieval_mode']
    threshold = config['threshold']
    use_retrieval = config['use_retrieval']
    randomly = config['randomly']
    index_base_path = config['index_base_path']


    # --- Load model + DB ---
    model, transform, non_empty_indices, index, original_reports, db_vectors = load_model_and_resources(
        base_path=index_base_path
    )

    gold_impression = []
    gold_findings = []
    samples_similar_reports = []
    predicted_labels = []

    count = 0
    for study_id, gt in study_ground_truth.items():
        gold_impression.append(gt["impression"])
        gold_findings.append(gt["findings"])
        # get txr vector
        txr_predicted = aggregated_scores[study_id]
        predicted_labels.append(txr_predicted)

        # --- Apply binarization for retrieval step ---
        if binarized_retrieval:
            pass
            # txr_for_retrieval = (txr_predicted >= threshold).astype(np.float32)
        else:
            txr_for_retrieval = txr_predicted

        if use_retrieval:
            if retrieval_mode == "whole":
                pass
            elif retrieval_mode == "partial":
                dict_symptom_reports = retrieve_for_top_symptoms(
                    txr_for_retrieval,
                    db_vectors,
                    original_reports,
                    top_k_symptoms=top_k_symptoms,
                    retrieved_k_reports=retrieved_k_reports,
                    randomly=randomly
                )
            else:
                raise Exception("retrieval_mode must be 'whole' or 'partial'")

            similar_reports = []
            for reps in dict_symptom_reports.values():
                similar_reports.extend(reps)
            samples_similar_reports.append(similar_reports)

        # create a dummy similar report for run_gpt_reporting_step function
        else: samples_similar_reports = [[] for _ in range(len(predicted_labels))]

        count += 1
        # if count >= 10:
        #     break

    # --- GPT reporting step ---
    run_gpt_reporting_step(
        config,
        predicted_labels,
        samples_similar_reports,
        model,
        gold_impression,
        gold_findings,
        threshold=threshold,
        retrieve_needed=use_retrieval,
        include_label_scores=include_label_scores
    )


# important booleans:
# retrieve_needed: whether use rag
# binarized: whether include probabilities
# randomly: whether use random documents instead of retrieving
def main():
    # all_experiments()
    # study_based_experiment()

    path = "configs/config.yaml"
    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    split = "test"
    # I decided to use mean, it is also possible to use max pool
    with open(f"./data/mean_scores_{split}.pkl", "rb") as f:
        # 3269 for test
        aggregated_scores = pickle.load(f)
    # a dict mapping study_id to ground_truth impression & findings
    # only for samples having both
    with open(f"./data/study_to_gold_{split}.pkl", "rb") as f:
        # 1624 for test
        study_ground_truth = pickle.load(f)

    run_experiment(config, aggregated_scores, study_ground_truth)
    impression_scores, findings_scores = evaluate_reports(config['output_path'])

    wandb.log({
        "impression": impression_scores,
        "findings": findings_scores,
    })

if __name__ == '__main__':
    main()
