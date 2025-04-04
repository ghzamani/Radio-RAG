import os
# todo check whether it works
os.environ['HF_HOME'] = "/mnt/disk2/ghazal.zamaninezhad/hf_cache"
# os.environ['HF_HOME'] = "/home/m_nobakhtian/mmed/hf_cache"

from transformers import AutoModelForCausalLM, AutoProcessor
from pathlib import Path
import torch
from datasets import load_dataset


def get_sample_bench():
    print("Loading chexpert dataset...")
    radiology_bench = load_dataset("ghazal-zamani/radiology_benchmark")['validation']
    # ch_plus = radiology_bench.filter(lambda x: x['dataset_name'] == 'chexpert_plus')
    ch_plus = radiology_bench
    print(f"Number of validation samples: {len(ch_plus)}")
    return ch_plus[0]


def load_maira_model(path, eval_mode=True):
    model = AutoModelForCausalLM.from_pretrained(path,
                                                 # torch_dtype=torch.float16,
                                                 trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)

    if eval_mode:
        model = model.eval()

    print("model loaded")
    return model, processor


def predict(model, processor, sample_data, device="cuda"):
    processed_inputs = processor.format_and_preprocess_reporting_input(
        current_frontal=sample_data["image"],
        current_lateral=None,
        prior_frontal=None,  # Our example has no prior
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
    print("Parsed prediction:", prediction)
    return prediction


def main():
    # load model
    path = "/mnt/disk2/ghazal.zamaninezhad/base_models/maira-2"
    # path = "microsoft/maira-2"
    model, processor = load_maira_model(path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    # get a sample of ch_plus
    ch_sample = get_sample_bench()

    prediction = predict(model, processor, ch_sample, device)
    print("original report:", ch_sample["report"])
if __name__ == "__main__":
    main()