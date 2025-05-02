import os
# os.environ['HF_HOME'] = "/home/m_nobakhtian/mmed/hf_cache"

import faiss
import torch, torchvision
import numpy as np
import torchxrayvision as xrv
import pickle
from tqdm import tqdm

from utills import load_test_bench, xray_transform, load_radio_bench


def save_to_database(vector, db_name):
    label_vectors = np.array(vector).astype('float32')  # FAISS needs float32
    # Normalize vectors if you want cosine similarity
    label_vectors = label_vectors / np.linalg.norm(label_vectors, axis=1, keepdims=True)
    # Build FAISS index
    d = label_vectors.shape[1]  # dimension
    index = faiss.IndexFlatIP(d)  # inner product = cosine similarity if normalized
    index.add(label_vectors)
    # Save FAISS index
    faiss.write_index(index, db_name)
    return index


def main():
    # load dataset
    dataset = load_test_bench()
    # dataset = load_radio_bench()
    # specify transform
    transform = torchvision.transforms.Compose(
        [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)])
    # load model
    model = xrv.models.DenseNet(weights="densenet121-res224-mimic_ch",
    # model = xrv.models.DenseNet(weights="densenet121-res224-chex",
                                cache_dir="/mnt/disk2/ghazal.zamaninezhad/hf_cache")
                                # cache_dir="/home/m_nobakhtian/mmed/hf_cache")
    # take model to gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # processed_images = []
    outputs = []
    reports = []
    for sample in tqdm(dataset['test'].select(range(100)), desc="Predicting"):
    # for sample in tqdm(dataset.select(range(100))):
        transformed = xray_transform(sample['image'], transform).to(device)
        # predict on dataset
        pred = model(transformed).flatten()
        outputs.append(pred.cpu().detach().numpy())
        reports.append(sample['report'])

    labels_vector = np.vstack(outputs)
    labels_index = save_to_database(labels_vector, "label_vector.index")

    with open("index_to_report.pkl", "wb") as f:
        pickle.dump(reports, f)


if __name__ == '__main__':
    main()
