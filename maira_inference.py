import os
# todo check whether it works
# os.environ['HF_HOME'] = "/mnt/disk2/ghazal.zamaninezhad/hf_cache"
os.environ['HF_HOME'] = "/home/m_nobakhtian/mmed/hf_cache"

from collections import defaultdict
import torch
from tqdm import tqdm

from inference import maira_predict
from utills import load_radio_bench, load_maira_model, save_list_to_txt


def separate_chexpert(ch_plus):
    """
    In chexpert dataset, there are some rows belonging to the same study.
    Separate those from the other and return both
    """
    # Count occurrences of each patient_id
    patient_counts = defaultdict(int)
    for row in ch_plus:
        patient_counts[row["patient_id"]] += 1
    # Get IDs that appear more than once
    duplicated_ids = [pid for pid, count in patient_counts.items() if count > 1]
    duplicates_dataset = ch_plus.filter(lambda x: x["patient_id"] in duplicated_ids)
    no_duplicates_dataset = ch_plus.filter(lambda x: x["patient_id"] not in duplicated_ids)
    return duplicates_dataset, no_duplicates_dataset

def group_multi_study_patient(multi_study_patients):
    """
    Gets the dataset of patients with multiple studies, groups them,
    and returns a list of tuples, indicating:
    (current_frontal_row, current_lateral_row, previous_frontal_row)
    Note that each item of tuple is a row of hf dataset
    """
    # Convert to pandas DataFrame
    duplicates_df = multi_study_patients.to_pandas()
    # Group by 'patient_id' (creates a DataFrameGroupBy object)
    grouped_duplicates = duplicates_df.groupby("patient_id")

    frontal_lateral = []
    # Iterate over all groups
    for patient_id, group in grouped_duplicates:
        # skip the row if it doesn't have findings
        if not group.iloc[0]['findings']:
            continue

        current_frontal_idx = None
        current_lateral_idx = None
        previous_frontal_idx = None

        if len(group) == 2:
            for row in group.itertuples():
                if row.frontal_lateral == 'Frontal':
                    current_frontal_idx = row.Index
                elif row.frontal_lateral == 'Lateral':
                    current_lateral_idx = row.Index
                else:
                    raise ValueError("value not defined")

        elif len(group) == 3:
            # I didn't find anything different between photos.
            # I think they were taken at the same time, but multiple photos.
            if group.iloc[-1].frontal_lateral == 'Frontal':
                current_frontal_idx = group.index[2].item()
                current_lateral_idx = group.index[1].item()
                previous_frontal_idx = group.index[0].item()
            else:
                current_frontal_idx = group.index[0].item()
                current_lateral_idx = group.index[1].item()
        else:
            raise ValueError("group not defined")

        current_frontal_row = multi_study_patients[current_frontal_idx]
        current_lateral_row = multi_study_patients[current_lateral_idx]
        previous_frontal_row = multi_study_patients[previous_frontal_idx] if previous_frontal_idx else None

        grouped_tuple = (current_frontal_row, current_lateral_row, previous_frontal_row)
        frontal_lateral.append(grouped_tuple)
    return frontal_lateral


def main():
    # load model
    # path = "/mnt/disk2/ghazal.zamaninezhad/base_models/maira-2"
    path = "microsoft/maira-2"
    model, processor = load_maira_model(path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    ch_plus = load_radio_bench()
    multi_study_patients, single_study_patients = separate_chexpert(ch_plus)
    # get list of frontal and lateral indices for multi study patients
    multi_study_grouped = group_multi_study_patient(multi_study_patients)

    # create a list of sample dictionaries
    samples = []
    # Process multi-study cases
    for curr_frontal, curr_lateral, prev_frontal in multi_study_grouped:
        samples.append({
            "current_frontal": curr_frontal['image'],
            "current_lateral": curr_lateral['image'],
            # when adding 3 images for one sample, got the error:
            # Token indices sequence length is longer than the specified maximum sequence length
            # "prior_frontal": prev_frontal['image'] if prev_frontal else None,
            "prior_frontal": None,
            "original_findings": curr_frontal['findings']
        })
        # break

    # Process single-study cases
    for patient in single_study_patients:
        # Skip if no findings or not a frontal image
        if not patient['findings'] or patient['frontal_lateral'] != 'Frontal':
            continue

        samples.append({
            "current_frontal": patient['image'],
            "current_lateral": None,
            "prior_frontal": None,
            "original_findings": patient['findings']
        })
        # break

    reference_findings = []
    predicted_findings = []
    for sample in tqdm(samples):
        prediction = maira_predict(model, processor, sample)
        predicted_findings.append(prediction)
        reference_findings.append(sample["original_findings"])

    assert len(predicted_findings) == len(samples)

    output_path = "/home/m_nobakhtian/mmed/Radio-RAG/outputs"
    # name format: {dataset}_{impression/findings}_model_{ref/pred}
    save_list_to_txt(predicted_findings, os.path.join(output_path,
                                                      'chplus_findings_maira_pred.txt'))
    save_list_to_txt(reference_findings, os.path.join(output_path,
                                                      'chplus_findings_maira_ref.txt'))

if __name__ == "__main__":
    main()