import faiss
import pickle
import torch, torchvision
import numpy as np
import torchxrayvision as xrv

from utills import load_test_bench, xray_transform


def retrieve_most_similar(predicted_label_vector, vector_index, reports, k=5):
    query_vector = np.array(predicted_label_vector).astype('float32')
    query_vector = query_vector / np.linalg.norm(query_vector)

    # Search
    _, indices = vector_index.search(query_vector.reshape(1, -1), k=k)  # get top 5 similar

    # Retrieve reports
    retrieved_reports = [reports[idx] for idx in indices[0]]
    return retrieved_reports


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
    for sample in samples:
        # predict labels for input image
        transformed = xray_transform(sample['image'], transform).to(device)
        # predict on dataset
        pred = model(transformed).flatten()
        pred = pred.cpu().detach().numpy()
        similar_reports = retrieve_most_similar(pred, index, reports, k=5)
        samples_similar_reports.append(similar_reports)


if __name__ == '__main__':
    main()
