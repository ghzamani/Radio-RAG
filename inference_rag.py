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


def build_prompt(label_vector, label_names, retrieved_reports, threshold=0.55):
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

    # Final prompt
    prompt = f"""You are a radiologist. Based on the following Findings and retrieved report excerpts, generate a radiology report that includes only the FINDINGS and IMPRESSION sections.

Write in a concise, professional tone as used in real chest X-ray reports. Do not include patient identifiers, clinical history, or template headers.

Findings:
{predicted_findings}
Retrieved similar reports:
{similar_reports}
Now write a new FINDINGS and IMPRESSION section for a similar case.
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
    # load vector database and reports
    index = faiss.read_index("label_vector.index")
    with open("index_to_report.pkl", "rb") as f:
        reports = pickle.load(f)

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
        similar_reports = retrieve_most_similar(pred, index, reports, k=5)
        samples_similar_reports.append(similar_reports)

    client = OpenAI(
        api_key=os.environ.get("API_KEY")
    )
    predicted_reports = []
    for labels, similar_reports in tqdm(zip(predicted_labels, samples_similar_reports),
                                        desc="Requesting GPT"):
        prompt = build_prompt(labels, model.pathologies, similar_reports)
        new_report = call_gpt(client, prompt)
        predicted_reports.append(new_report)
        # break

    save_result(gold_impression, "gold_impressions.jsonl")
    save_result(gold_findings, "gold_findings.jsonl")
    save_result(predicted_reports, "predicted_reports.jsonl")

if __name__ == '__main__':
    main()
