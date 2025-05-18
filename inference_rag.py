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

from utills import load_test_bench, xray_transform, load_radio_bench, extract_sections


def retrieve_most_similar(predicted_label_vector, vector_index, reports, k=5):
    query_vector = np.array(predicted_label_vector).astype('float32')
    query_vector = query_vector / np.linalg.norm(query_vector)

    # Search
    _, indices = vector_index.search(query_vector.reshape(1, -1), k=k)  # get top 5 similar

    # Retrieve reports
    retrieved_reports = [reports[idx] for idx in indices[0]]
    return retrieved_reports


def build_prompt(label_vector, label_names, retrieved_reports, threshold=0.55):
    # Convert label vector to readable findings
    predicted_findings = ""
    label_list = label_vector.tolist()
    for value, name in zip(label_list, label_names):
        if not name:
            continue
        # check if value is more than threshold
        if value >= threshold:
            predicted_findings += f"- {name}: {round(value, 2)}\n"
    # TODO what if all labels where under threshold? predicted_findings would be empty
    if not predicted_findings:
        predicted_findings = "None"

    similar_reports = ""
    # Add retrieved reports
    for i, report in enumerate(retrieved_reports):
        # findings, impression = extract_sections(report)
        # if not findings and not impression:
        #     print("The report has neither findings nor impression")
        #     continue
        similar_reports += f"\n--- Report {i+1} ---\n"
        # if findings:
        #     similar_reports += f"FINDINGS:\n{findings}\n"
        # if impression:
        #     similar_reports += f"IMPRESSION:\n{impression}\n"
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


def main():
    # radio_bench_val = load_test_bench()
    # radio_bench_val = load_radio_bench()
    radio_bench_val = load_dataset("/mnt/disk2/ghazal.zamaninezhad/data/mimic_radio")['validation']
    # take first 5 examples
    # samples = radio_bench_val['validation'].select(range(5))
    samples = radio_bench_val.select(range(300, 310))
    # specify transform
    transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),
                                                xrv.datasets.XRayResizer(224)])
    # load model
    model = xrv.models.DenseNet(weights="densenet121-res224-mimic_ch",
    # model = xrv.models.DenseNet(weights="densenet121-res224-chex",
                                cache_dir="/mnt/disk2/ghazal.zamaninezhad/hf_cache")
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
        api_key=os.environ.get("API_KEY"),
        # timeout = 20.0
    )
    predicted_reports = []
    for labels, similar_reports in tqdm(zip(predicted_labels, samples_similar_reports),
                                        desc="Requesting GPT"):
        prompt = build_prompt(labels, model.pathologies, similar_reports)
        print(prompt)
        # print("Requesting GPT ...")
        new_report = call_gpt(client, prompt)
        predicted_reports.append(new_report)
        print(new_report)
        break


if __name__ == '__main__':
    main()
