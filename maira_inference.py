import os
# todo check whether it works
# os.environ['HF_HOME'] = "/mnt/disk2/ghazal.zamaninezhad/hf_cache"
os.environ['HF_HOME'] = "/home/m_nobakhtian/mmed/hf_cache"

from transformers import AutoModelForCausalLM, AutoProcessor
import torch
from datasets import load_dataset
from collections import defaultdict
import pandas as pd


def get_radio_bench():
    print("Loading chexpert dataset...")
    # radiology_bench = load_dataset("ghazal-zamani/radiology_benchmark")['validation']
    radiology_bench = load_dataset("ghazal-zamani/radio_benchmark")['validation']
    # ch_plus = radiology_bench.filter(lambda x: x['dataset_name'] == 'chexpert_plus')
    ch_plus = radiology_bench
    print(f"Number of validation samples: {len(ch_plus)}")
    # return ch_plus[0]
    return ch_plus

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


def load_maira_model(path, eval_mode=True):
    model = AutoModelForCausalLM.from_pretrained(path,
                                                 torch_dtype=torch.float16,
                                                 trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)

    if eval_mode:
        model = model.eval()

    print("model loaded")
    return model, processor


def predict(model, processor, sample_data, device="cuda"):
    processed_inputs = processor.format_and_preprocess_reporting_input(
        current_frontal=sample_data["current_frontal"],
        current_lateral=sample_data["current_lateral"],
        prior_frontal=sample_data["prior_frontal"],
        indication=None,
        technique=None,
        comparison=None,
        prior_report=None,  # Our example has no prior
        return_tensors="pt",
        get_grounding=False,  # For this example we generate a non-grounded report
    )

    processed_inputs = processed_inputs.to(device)
    with torch.no_grad():
        output_decoding = model.generate(
            **processed_inputs,
            max_new_tokens=300,  # Set to 450 for grounded reporting
            use_cache=True,
        )
    prompt_length = processed_inputs["input_ids"].shape[-1]
    decoded_text = processor.decode(output_decoding[0][prompt_length:], skip_special_tokens=True)
    decoded_text = decoded_text.lstrip()  # Findings generation completions have a single leading space
    prediction = processor.convert_output_to_plaintext_or_grounded_sequence(decoded_text)
    # print("Parsed prediction:", prediction)
    return prediction


def main():
    # load model
    # path = "/mnt/disk2/ghazal.zamaninezhad/base_models/maira-2"
    path = "microsoft/maira-2"
    model, processor = load_maira_model(path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    ch_plus = get_radio_bench()
    multi_study_patients, single_study_patients = separate_chexpert(ch_plus)
    # get list of frontal and lateral indices for multi study patients
    multi_study_grouped = group_multi_study_patient(multi_study_patients)

    reference_findings = []
    predicted_findings = []
    # create a list of sample dictionaries
    samples = []
    # Process multi-study cases
    for curr_frontal, curr_lateral, prev_frontal in multi_study_grouped:
        samples.append({
            "current_frontal": curr_frontal['image'],
            "current_lateral": curr_lateral['image'],
            "prior_frontal": prev_frontal['image'] if prev_frontal else None,
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

    for sample in samples:
        prediction = predict(model, processor, sample)
        predicted_findings.append(prediction)
        reference_findings.append(sample["original_findings"])

    print(reference_findings)
    print(predicted_findings)

if __name__ == "__main__":
    main()