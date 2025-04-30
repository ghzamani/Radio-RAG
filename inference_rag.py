import faiss
import pickle
import torch, torchvision
import numpy as np
import torchxrayvision as xrv
import openai

from utills import load_test_bench, xray_transform


def retrieve_most_similar(predicted_label_vector, vector_index, reports, k=5):
    query_vector = np.array(predicted_label_vector).astype('float32')
    query_vector = query_vector / np.linalg.norm(query_vector)

    # Search
    _, indices = vector_index.search(query_vector.reshape(1, -1), k=k)  # get top 5 similar

    # Retrieve reports
    retrieved_reports = [reports[idx] for idx in indices[0]]
    return retrieved_reports


def build_prompt(label_vector, label_names, retrieved_reports, threshold=0.5):
    # Convert label vector to readable findings
    findings = "\n".join(
        f"- {label_names[i]}: {round(val, 2)}"
        for i, val in enumerate(label_vector[0])
        if val >= threshold
    )

    # Add retrieved reports
    similar_reports = "\n".join(
        f"Report {i+1}:\n{report}" for i, report in enumerate(retrieved_reports)
    )

    # Final prompt
    prompt = f"""You are a radiologist. Based on the following predicted findings and similar historical reports, write a new radiology report.

Predicted findings:
{findings}

Retrieved similar reports:
{similar_reports}

Write the new report below:
Report: 

"""
    return prompt


def call_gpt(prompt, model="gpt-4o-mini"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system",
             "content": "You are an expert radiologist generating chest X-ray reports."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=500
    )
    return response['choices'][0]['message']['content']


def main():
    radio_bench_val = load_test_bench()
    # take first 5 examples
    samples = radio_bench_val['validation'].select(range(5))
    # specify transform
    transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),
                                                xrv.datasets.XRayResizer(224)])
    # load model
    # model = xrv.models.DenseNet(weights="densenet121-res224-mimic_ch",
    model = xrv.models.DenseNet(weights="densenet121-res224-chex",
                                cache_dir="/mnt/disk2/ghazal.zamaninezhad/hf_cache")
    # take model to device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    # load vector database and reports
    index = faiss.read_index("label_vector.index")
    with open("index_to_report.pkl", "rb") as f:
        reports = pickle.load(f)

    samples_similar_reports = []
    predicted_labels = []
    for sample in samples:
        # predict labels for input image
        transformed = xray_transform(sample['image'], transform).to(device)
        # predict on dataset
        pred = model(transformed).flatten()
        pred = pred.cpu().detach().numpy()
        predicted_labels.append(pred)
        similar_reports = retrieve_most_similar(pred, index, reports, k=5)
        samples_similar_reports.append(similar_reports)


    openai.api_key = "dummy"
    for labels, similar_reports in zip(predicted_labels, samples_similar_reports):
        prompt = build_prompt(labels, model.pathologies, similar_reports)
        print(prompt)
        # new_report = call_gpt(prompt)
        # print(new_report)
        break


if __name__ == '__main__':
    main()
