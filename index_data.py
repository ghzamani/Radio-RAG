import os
# os.environ['HF_HOME'] = "/home/m_nobakhtian/mmed/hf_cache"

import faiss
import torch, torchvision
import numpy as np
import torchxrayvision as xrv
import pickle
from tqdm import tqdm
import pandas as pd

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


def find_studies_path(root_dir):
    studies_names = []
    images_path = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if not dirnames:  # If no subdirectories, it's a last folder
            # add study name to a list
            study_name = int(dirpath.split('/')[-1][1:])
            studies_names.append(study_name)
            # add images path to a list
            images = [os.path.join(dirpath, file_path) for file_path in filenames if file_path.endswith(".jpg")]
            if images:  # Only add if folder contains files
                images_path.extend(images)
    return studies_names, images_path


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
    # main()
    studies_names, images = find_studies_path("/mnt/disk2/ghazal.zamaninezhad/mimic/files/mimic-cxr-jpg/2.1.0/files/p15")
    # read all the reports
    report_sections = pd.read_csv("/mnt/disk2/ghazal.zamaninezhad/mimic/files/mimic-cxr-jpg/2.1.0/files/mimic_cxr_sectioned.csv")
    # filter those reports which has both findings and impression
    non_null_sections = report_sections[report_sections['findings'].notnull() & report_sections['impression'].notnull()]
    # toke rows that belong to train data
    related_non_null = non_null_sections[non_null_sections['study_id'].isin(studies_names)]
    print(len(related_non_null))
