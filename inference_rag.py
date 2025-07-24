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


def build_prompt(label_vector, label_names, retrieved_reports, threshold=0.55, retrieved=True):
    # Convert label vector to readable findings
    label_list = label_vector.tolist()
    # Filter out None names and create valid pairs
    valid_pairs = [(v, n) for v, n in zip(label_list, label_names) if n]
    # Get items above threshold
    above_threshold = [f"- {n}: {round(v, 2)}" for v, n in valid_pairs if v >= threshold]

    if above_threshold:
        predicted_findings = "\n".join(above_threshold)

    # TODO what if all labels where under threshold? predicted_findings would be empty
    # I added top 3 symptoms, just not to have empty findings
    else:
        # Get top 3 and format
        top3 = sorted(valid_pairs, key=lambda x: x[0], reverse=True)[:3]
        predicted_findings = "\n".join(f"- {n}: {round(v, 2)}" for v, n in top3)

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
    with open(file_path, "w") as f:
        for idx, item in enumerate(input_list):
            data = {idx: item}
            f.write(json.dumps(data) + '\n')

def main():
    # radio_bench_val = load_test_bench()
    # radio_bench_val = load_radio_bench()
    # radio_bench_val = load_dataset("/mnt/disk2/ghazal.zamaninezhad/data/mimic_radio")['validation']
    radio_bench_val = load_dataset("ghazal-zamani/mimic_radio")['validation']
    # TODO only take samples with both impression and findings? in order to evaluate
    # coulnd't apply filter because of low RAM
    # radio_bench_val = radio_bench_val.filter(non_null_finding_impression)
    samples = []
    gold_impression = []
    gold_findings = []
    for s in tqdm(radio_bench_val):
        if s['impression'] and s['findings']:
            samples.append(s)
            gold_impression.append(s['impression'])
            gold_findings.append(s['findings'])
        if len(samples) == 50:
            break

    # specify transform
    transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),
                                                xrv.datasets.XRayResizer(224)])
    # load model
    model = xrv.models.DenseNet(weights="densenet121-res224-mimic_ch")
    # model = xrv.models.DenseNet(weights="densenet121-res224-chex",
    #                             cache_dir="/mnt/disk2/ghazal.zamaninezhad/hf_cache")
                                # cache_dir="/home/m_nobakhtian/mmed/hf_cache")
    # take model to device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    # load vector database
    # index = faiss.read_index("label_vector.index")
    # load reports
    with open("index_to_report.pkl", "rb") as f:
        original_reports = pickle.load(f)
    # load original vectors saved in db
    db_vectors = np.load("symptoms_vectors.npy")

    # find indices of pathologies (11 out of 18)
    non_empty_indices = [i for i, name in enumerate(model.pathologies) if name]
    samples_similar_reports = []
    predicted_labels = []
    for sample in tqdm(samples,
                       desc="Predicting pathologies on test data"):
        # predict labels for input image
        transformed = xray_transform(sample['image'], transform).to(device)
        # predict on dataset
        pred = model(transformed).flatten()
        pred = pred.cpu().detach().numpy()
        # Filter the vector using these indices
        pred = pred[non_empty_indices]
        predicted_labels.append(pred)
        # similar_reports = retrieve_most_similar(pred, index, original_reports, k=5)
        # add no retrieved report to check whether rag has any effect
        # TODO IMPORTANT
        dict_symptom_reports = retrieve_for_top_symptoms(pred,
                                                        db_vectors,
                                                        original_reports,
                                                        top_k_symptoms=2,
                                                        retrieved_k_reports=3,
                                                        randomly=True)
        similar_reports = []
        for reps in dict_symptom_reports.values():
            similar_reports.extend(reps)
        samples_similar_reports.append(similar_reports)

    client = OpenAI(
        api_key=os.environ.get("API_KEY")
    )
    predicted_reports = []
    for labels, similar_reports in tqdm(zip(predicted_labels, samples_similar_reports),
                                        desc="Requesting GPT"):
        # TODO IMPORTANT!
        prompt = build_prompt(labels, model.pathologies, similar_reports)
        # prompt = build_prompt(labels, model.pathologies, similar_reports, retrieved=False)
        new_report = call_gpt(client, prompt)
        predicted_reports.append(new_report)
        # break
    # dir_name = "exp2_no_rag/ "
    dir_name = "exp3/ "
    save_result(gold_impression, dir_name + "gold_impressions.jsonl")
    save_result(gold_findings, dir_name + "gold_findings.jsonl")
    save_result(predicted_reports, dir_name + "predicted_reports.jsonl")

if __name__ == '__main__':
    main()
