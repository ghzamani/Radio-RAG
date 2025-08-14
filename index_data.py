import os
# os.environ['HF_HOME'] = "/home/m_nobakhtian/mmed/hf_cache"

import faiss
import torch, torchvision
import numpy as np
import torchxrayvision as xrv
import pickle
from tqdm import tqdm
import pandas as pd
import skimage
from collections import defaultdict
import glob
import re

from utills import xray_transform, clean_report

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def format_report(row, add_delimiters=True):
    # I forgot to clean them before writing to index.
    # so do it later with clean_report function
    if add_delimiters:
        return f"FINDINGS: {row['findings'].strip()} \n IMPRESSION: {row['impression'].strip()}"
    return row['findings'].strip() + row['impression'].strip()


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
    # meta_important_views = meta_sorted.drop_duplicates(subset="study_id", keep="first")

    # only keep train rows from meta dataframe
    # df_train = meta_important_views[meta_important_views["study_id"].isin(train_unique_studies)]
    df_train = meta_sorted[meta_sorted["study_id"].isin(train_unique_studies)]
    print(f"total train samples: {len(df_train)}")
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
    # full_text = report_sections['findings'] + report_sections['impression']
    report_sections['report'] = report_sections.apply(format_report, axis=1)
    # add reports to dataframe
    # count = 232855
    merged_df = pd.merge(df_train, report_sections, on="study_id", how="inner")
    print(f"total train samples with both imp & fin: {len(merged_df)}")
    return merged_df[['study_id', 'ViewPosition', 'report', 'image_path']]
    # I guess in the end we should have 125417 (train studies with both i,f)


def save_partial_results(embed_dict, report_dict, chunk_id, out_dir='./embeddings'):
    with open(f"{out_dir}/embed_dict_{chunk_id}.pkl", "wb") as f:
        pickle.dump(embed_dict, f)
    with open(f"{out_dir}/report_dict_{chunk_id}.pkl", "wb") as f:
        pickle.dump(report_dict, f)
    # np.save(f"{out_dir}/pooled_vectors_{chunk_id}.npy", pooled_vectors)
    print(f"[Saved at step {chunk_id}]")


def predict_txr_study_based(model, transform, df, images_path, label_idx,
                            chunk_size=500, out_dir="embeddings"):
    os.makedirs(out_dir, exist_ok=True)
    study_embeddings = defaultdict(dict)
    study_reports = {}


    # chunk_id = 0
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Embedding images"):
        study_id = row["study_id"]
        view = row["ViewPosition"]
        report = row["report"]
        full_path = os.path.join(images_path, row["image_path"])

        try:
            img = skimage.io.imread(full_path)
        except Exception as e:
            print(f"Failed for {full_path}: {e}")
            continue

        transformed = xray_transform(img, transform).to(DEVICE)
        with torch.no_grad():
            scores = model(transformed).flatten()
        scores = scores[label_idx]

        study_embeddings[study_id][view] = scores.cpu()
        study_reports[study_id] = report

        if (i + 1) % chunk_size == 0:
            save_partial_results(dict(study_embeddings),
                                 study_reports,
                                 i,
                                 out_dir)
            # empty ram
            study_embeddings.clear()
            study_reports.clear()
            # chunk_id += 1

    if study_embeddings:
        save_partial_results(dict(study_embeddings),
                             study_reports,
                             i,
                             out_dir)


def aggregate_and_save_to_faiss(embed_dir, out_path):
    print("Start aggregating results")
    all_study_views = defaultdict(dict)

    # Load chunked embedding dicts
    for embed_file in sorted(glob.glob(f"{embed_dir}/embed_dict_*.pkl")):
        with open(embed_file, "rb") as f:
            chunk = pickle.load(f)
            for study_id, view_dict in chunk.items():
                for view, tensor in view_dict.items():
                    if view in all_study_views[study_id]:
                        print(
                            f"Duplicate view '{view}' found for study_id '{study_id}' in chunk '{embed_file}'")
                    all_study_views[study_id][view] = tensor  # Overwrite if duplicate

    all_embeddings = []
    study_ids = []
    all_reports = []
    for study_id, view_dict in all_study_views.items():
        view_tensors = torch.stack(list(view_dict.values()))
        # mean got better result than max, so replace
        # pooled = torch.max(view_tensors, dim=0).values.numpy()
        pooled = torch.mean(view_tensors, dim=0).numpy()
        all_embeddings.append(pooled)
        study_ids.append(study_id)

    # Collect reports in the same order
    report_lookup = {}
    for report_file in sorted(glob.glob(f"{embed_dir}/report_dict_*.pkl")):
        with open(report_file, "rb") as f:
            report_lookup.update(pickle.load(f))
    for sid in study_ids:
        try:
            rep = report_lookup[sid]
            # pre-process reports
            rep = clean_report(rep)
        except KeyError:
            print(f"{sid} not found in report_lookup")
            rep = ""
        all_reports.append(rep)

    all_embeddings = np.vstack(all_embeddings).astype("float32")
    print("saving l1")
    # L1 index
    index_l1 = faiss.IndexFlat(all_embeddings.shape[1], faiss.METRIC_L1)
    index_l1.add(all_embeddings)
    faiss.write_index(index_l1, f"{out_path}/l1.index")

    print("saving l2")
    # L2 index
    index_l2 = faiss.IndexFlatL2(all_embeddings.shape[1])
    index_l2.add(all_embeddings)
    faiss.write_index(index_l2, f"{out_path}/l2.index")

    print("saving cosine")
    # Cosine similarity index (normalize first)
    norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    normed_embeddings = all_embeddings / np.maximum(norms, 1e-10)
    index_cos = faiss.IndexFlatIP(all_embeddings.shape[1])  # inner product == cosine if normalized
    index_cos.add(normed_embeddings)
    faiss.write_index(index_cos, f"{out_path}/cosine.index")


    with open(f"{out_path}/index_to_report.pkl", "wb") as f:
        pickle.dump(all_reports, f)
    # consistency between indices and study ids
    # I think that wouldn't be used
    with open(f"{out_path}/index_to_study_id.pkl", "wb") as f:
        pickle.dump(study_ids, f)
    np.save(f"{out_path}/symptoms_vectors.npy", all_embeddings)


def main():
    # data_path = "/mnt/disk2/ghazal.zamaninezhad/codes/Radio-RAG/data/"
    data_path = "/root/codes/Radio-RAG/data/"
    splits_path = data_path + "mimic-cxr-2.0.0-split.csv"
    meta_path = data_path + "mimic-cxr-2.0.0-metadata.csv"
    reports_path = data_path + "mimic_cxr_sectioned.csv"
    # images_root = "/volumes/hetzner/zamaninezhad/my_data/physionet.org/files/mimic-cxr-jpg/2.1.0/"
    images_root = "/mnt/hetzner/zamaninezhad/my_data/physionet.org/files/mimic-cxr-jpg/2.1.0/"

    splits = pd.read_csv(splits_path)
    metadata = pd.read_csv(meta_path)
    report_sections = pd.read_csv(reports_path)

    study_img_report = map_study_to_best_image(splits, metadata, report_sections)

    # specify transform
    transform = torchvision.transforms.Compose(
        [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)])
    # load model
    model = xrv.models.DenseNet(weights="densenet121-res224-mimic_ch")
    # take model to gpu
    model = model.to(DEVICE)
    model.eval()

    # check which symptoms relate to this dataset
    # find indices of pathologies (11 out of 18)
    valid_indices = [i for i, name in enumerate(model.pathologies) if name]

    predict_txr_study_based(model, transform, study_img_report, images_root, valid_indices,
                            chunk_size=500, out_dir="embeddings")
    aggregate_and_save_to_faiss("embeddings", "embeddings")


if __name__ == '__main__':
    # main()
    aggregate_and_save_to_faiss("/root/data/embeddings",
                                "/root/data/train_index")
