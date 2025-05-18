import os
# os.environ['HF_HOME'] = "/home/m_nobakhtian/mmed/hf_cache"

import faiss
import torch, torchvision
import numpy as np
import torchxrayvision as xrv
import pickle
from tqdm import tqdm
import pandas as pd
import random
import skimage

# from utills import load_test_bench, xray_transform, load_radio_bench


def xray_transform(img, transform):
    # Prepare the image:
    img = np.array(img)
    img = xrv.datasets.normalize(img, 255)  # convert 8-bit image to [-1024, 1024] range
    # TODO ? handle RGB images (as it is in their repo)
    img = img[None, ...]

    img = transform(img)
    img = torch.from_numpy(img)
    return img


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


def map_study_to_best_image(splits, metadata, report_sections):
    view_position_mapping = {
        "PA": 1,
        "AP": 2,
        "LATERAL": 3,
        "LL": 4,
        "AP AXIAL": 5,
        "AP LLD": 6,
        "AP RLD": 7,
        "PA RLD": 8,
        "PA LLD": 9,
        "LAO": 10,
        "RAO": 11,
        "LPO": 12,
        "XTABLE LATERAL": 13,
        "SWIMMERS": 14,
        "": 15  # Empty string maps to 15
    }

    # get train study names
    # count = 222758
    train_unique_studies = splits[splits['split']=='train']['study_id'].unique()

    # image names are unique (checked)
    # map each view to its priority
    metadata['view_priority'] = metadata["ViewPosition"].map(view_position_mapping)
    # sort based on priority
    meta_sorted = metadata.sort_values(by=["study_id", "view_priority"])
    # keep the most important view for each study
    # count = 227835
    meta_important_views = meta_sorted.drop_duplicates(subset="study_id", keep="first")

    # only keep train rows from meta dataframe
    df_train = meta_important_views[meta_important_views["study_id"].isin(train_unique_studies)]
    df_train['image_path'] = (
            'files/p' + df_train['subject_id'].astype(str).str[:2] + '/' +  # p10/
            'p' + df_train['subject_id'].astype(str) + '/' +  # p10003502/
            's' + df_train['study_id'].astype(str) + '/' +  # s50084553/
            df_train['dicom_id'].astype(str) + '.jpg'  # 70d7e600-....jpg
    )
    # omit rows which don't have either impression or findings
    # count after omitting = 128032
    report_sections = report_sections[report_sections['impression'].notnull() & report_sections['findings'].notnull()]
    # concat findings and impression
    # todo add impression and findings keyword before each?
    report_sections['report'] = report_sections['findings'] + report_sections['impression']
    # add reports to dataframe
    # count = 125417
    merged_df = pd.merge(df_train, report_sections, on="study_id", how="inner")
    return merged_df[['study_id', 'report', 'image_path']]


def main():
    data_path = "/mnt/disk2/ghazal.zamaninezhad/codes/Radio-RAG/data/"
    splits_path = data_path + "mimic-cxr-2.0.0-split.csv"
    meta_path = data_path + "mimic-cxr-2.0.0-metadata.csv"
    reports_path = data_path + "mimic_cxr_sectioned.csv"
    images_path = "/volumes/hetzner/zamaninezhad/my_data/physionet.org/files/mimic-cxr-jpg/2.1.0/"
    # images_path = "/mnt/hetzner/zamaninezhad/my_data/physionet.org/files/mimic-cxr-jpg/2.1.0/"

    splits = pd.read_csv(splits_path)
    metadata = pd.read_csv(meta_path)
    report_sections = pd.read_csv(reports_path)

    study_img_report = map_study_to_best_image(splits, metadata, report_sections)
    # select 100 random rows from train data
    random_study_img_report = study_img_report.sample(n=100, random_state=42)

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

    # check which symptoms relate to this dataset
    # find indices of pathologies (11 out of 18)
    non_empty_indices = [i for i, name in enumerate(model.pathologies) if name]
    outputs = []
    reports = []
    for index, sample in tqdm(random_study_img_report.iterrows(),
                           total=len(random_study_img_report),
                           desc="Processing rows"):
        full_path = images_path + sample['image_path']
        img = skimage.io.imread(full_path)
        transformed = xray_transform(img, transform).to(device)
        # predict on dataset
        pred = model(transformed).flatten()
        # Filter the vector using these indices
        pred = pred[non_empty_indices]
        outputs.append(pred.cpu().detach().numpy())
        reports.append(sample['report'])

    labels_vector = np.vstack(outputs)
    labels_index = save_to_database(labels_vector, "label_vector.index")

    with open("index_to_report.pkl", "wb") as f:
        pickle.dump(reports, f)


if __name__ == '__main__':
    main()
