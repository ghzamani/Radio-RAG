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
# from utills import load_test_bench, xray_transform, load_radio_bench, extract_sections
from utills import xray_transform

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


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


def build_prompt(label_vector, label_names, retrieved_reports,
                 threshold=0.55, retrieved=True, binarized=False):
    # Convert label vector to readable findings
    label_list = label_vector.tolist()
    # Filter out None names and create valid pairs
    valid_pairs = [(v, n) for v, n in zip(label_list, label_names) if n]
    # Get items above threshold
    if not binarized:
        above_threshold = [f"- {n}: {round(v, 2)}" for v, n in valid_pairs if v >= threshold]
    # only add names, not values
    else:
        above_threshold = [f"- {n}" for v, n in valid_pairs if v >= threshold]

    if above_threshold:
        predicted_findings = "\n".join(above_threshold)

    # TODO what if all labels where under threshold? predicted_findings would be empty
    # I added top 3 symptoms, just not to have empty findings
    else:
        # Get top 3 and format
        top3 = sorted(valid_pairs, key=lambda x: x[0], reverse=True)[:3]
        if binarized:
            predicted_findings = "\n".join(f"- {n}: {round(v, 2)}" for v, n in top3)
        else:
            predicted_findings = "\n".join(f"- {n}" for _, n in top3)

    similar_reports = ""
    # Add retrieved reports
    for i, report in enumerate(retrieved_reports):
        similar_reports += f"\n--- Report {i+1} ---\n"
        similar_reports += f"{report}\n"

    if retrieved:
        # Final prompt
        prompt = f"""You are a radiologist. Based on the following Findings and retrieved report excerpts, generate a radiology report that includes only the FINDINGS and IMPRESSION sections.

Write in a concise, professional tone as used in real chest X-ray reports. Do not include patient identifiers, clinical history, or template headers.

Findings:
{predicted_findings}

Retrieved similar reports:
{similar_reports}
Now write a new FINDINGS and IMPRESSION section for a similar case.
"""
    else:
        prompt = f"""You are a radiologist. Based on the following Findings, generate a radiology report that includes only the FINDINGS and IMPRESSION sections.

Write in a concise, professional tone as used in real chest X-ray reports. Do not include patient identifiers, clinical history, or template headers.

Findings:
{predicted_findings}

Now write a new FINDINGS and IMPRESSION section for this case.
"""
    return prompt


def call_gpt(client, prompt, model="gpt-4o-mini"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system",
             "content": "You are a radiologist assistant generating accurate and concise chest X-ray reports."},
            {"role": "user",
             "content": prompt}
        ],
        temperature=0,
        max_tokens=300
    )
    return response.choices[0].message.content

def non_null_finding_impression(row):
    return row['findings'] and row['impression']

def save_result(input_list, file_path):
    # todo create directory if doesn't exist
    with open(file_path, "w") as f:
        for idx, item in enumerate(input_list):
            data = {idx: item}
            f.write(json.dumps(data) + '\n')


def load_and_prepare_samples(max_samples=50, split='validation'):
    assert split in ['validation', 'test']

    radio_bench_val = load_dataset("ghazal-zamani/mimic_radio")[split]
    samples, gold_impression, gold_findings = [], [], []
    # TODO only take samples with both impression and findings? in order to evaluate
    # couldn't apply filter because of low RAM
    # radio_bench_val = radio_bench_val.filter(non_null_finding_impression)
    for s in tqdm(radio_bench_val, desc="Filtering samples"):
        if s['impression'] and s['findings']:
            samples.append(s)
            gold_impression.append(s['impression'])
            gold_findings.append(s['findings'])
        if len(samples) == max_samples:
            break
    return samples, gold_impression, gold_findings


def load_model_and_resources(base_path=''):
    transform = torchvision.transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224)
    ])
    model = xrv.models.DenseNet(weights="densenet121-res224-mimic_ch")
    model = model.to(DEVICE)

    index = faiss.read_index(f"{base_path}/label_vector.index")
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

def run_gpt_reporting_step(predicted_labels,
                           samples_similar_reports,
                           model,
                           gold_impression,
                           gold_findings,
                           dir_name,
                           threshold=0.55,
                           retrieve_needed=True,
                           binarized=False):
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
        # api_key=os.environ.get("API_KEY")
    )
    predicted_reports = []

    for labels, similar_reports in tqdm(zip(predicted_labels, samples_similar_reports),
                                        desc="Requesting GPT"):
        prompt = build_prompt(labels, model.pathologies, similar_reports,
                              threshold=threshold,
                              retrieved=retrieve_needed,
                              binarized=binarized)
        new_report = call_gpt(client, prompt)
        predicted_reports.append(new_report)
        # break  # for debugging

    save_result(gold_impression, dir_name + "gold_impressions.jsonl")
    save_result(gold_findings, dir_name + "gold_findings.jsonl")
    save_result(predicted_reports, dir_name + "predicted_reports.jsonl")


def all_experiments():
    all_experiment_modes = ['no_rag', 'rag', 'random_rag']
    experiment_mode = 'no_rag'
    assert experiment_mode in all_experiment_modes

    retrieve_needed = True
    random_documents = False

    if experiment_mode == 'no_rag':
        retrieve_needed = False
    if experiment_mode == 'random_rag':
        random_documents = True

    # dir_name = "exp3/ "
    dir_name = "test/ "
    # load samples and ground truth
    samples, gold_impression, gold_findings = load_and_prepare_samples(max_samples=50)
    # load model and vector database
    model, transform, non_empty_indices, index, original_reports, db_vectors = load_model_and_resources(base_path='./embeddings')
    # predict txr on each sample and retrieve documents
    predicted_labels, samples_similar_reports = predict_retrieve(model, transform, samples, db_vectors, original_reports,
                                                                 top_k_symptoms=2,
                                                                 retrieved_k_reports=3,
                                                                 randomly=random_documents)

    # TODO pay attention to retrieved boolean!
    run_gpt_reporting_step(predicted_labels, samples_similar_reports,
                           model, gold_impression, gold_findings,
                           dir_name, retrieve_needed)

def study_based_experiment():
    """
    I created a dict mapping study_id -> gold_impression, gold_findings in colab
    for those samples which have both impression and findings
    also, we have a dict mapping study_id -> txr vector (max values)

    - read both dicts
    - iterate through first one, find it in the second
    - send request to gpt
    """

    split = "val"
    with open(f"./data/aggregated_scores_{split}.pkl", "rb") as f:
        aggregated_scores = pickle.load(f)
        # we have 1808 studies (2991 total images)

    # a dict mapping study_id to ground_truth impression & findings
    # only for samples having both
    with open(f"./data/study_to_gold_{split}.pkl", "rb") as f:
        study_ground_truth = pickle.load(f)
        # we have 991 studies with both impression and findings
        # load model and vector database
    model, transform, non_empty_indices, index, original_reports, db_vectors = load_model_and_resources(
        base_path='./embeddings')

    gold_impression = []
    gold_findings = []
    samples_similar_reports = []
    predicted_labels = []

    count = 0
    for study_id, gt in study_ground_truth.items():
        gold_impression.append(gt['impression'])
        gold_findings.append(gt['findings'])
        # get the txr vector
        txr_predicted = aggregated_scores[study_id]
        # retrieve related docs for each
        dict_symptom_reports = retrieve_for_top_symptoms(txr_predicted,
                                                         db_vectors,
                                                         original_reports,
                                                         top_k_symptoms=2,
                                                         retrieved_k_reports=3,
                                                         randomly=False)
        similar_reports = []
        for reps in dict_symptom_reports.values():
            similar_reports.extend(reps)
        samples_similar_reports.append(similar_reports)
        predicted_labels.append(txr_predicted)
        count += 1
        if count == 50:
            break
    # using all train data in index
    dir_name = 'study_exp6/'
    retrieve_needed = True
    binarized = False
    # TODO pay attention to retrieved boolean!
    run_gpt_reporting_step(predicted_labels, samples_similar_reports,
                           model, gold_impression, gold_findings,
                           dir_name,
                           threshold=0.7, retrieve_needed=retrieve_needed, binarized=binarized)

# important booleans:
# retrieve_needed: whether use rag
# binarized: whether include probabilities
# randomly: whether use random documents instead of retrieving

def main():
    # all_experiments()
    study_based_experiment()

if __name__ == '__main__':
    main()
