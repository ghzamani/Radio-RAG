import json
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoProcessor,
                          BertTokenizer, ViTImageProcessor,
                          VisionEncoderDecoderModel)
import numpy as np
import torchxrayvision as xrv
import torch
import re

def load_chexpert_model(model_name_or_path="IAMJB/chexpert-mimic-cxr-impression-baseline",
                        eval_mode=True):
    model = VisionEncoderDecoderModel.from_pretrained(model_name_or_path)
    if eval_mode:
        model = model.eval()
    tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
    image_processor = ViTImageProcessor.from_pretrained(model_name_or_path)

    generation_args = {
        "bos_token_id": model.config.bos_token_id,
        "eos_token_id": model.config.eos_token_id,
        "pad_token_id": model.config.pad_token_id,
        "num_return_sequences": 1,
        "max_length": 128,
        "use_cache": True,
        "beam_width": 2,
    }
    return model, tokenizer, image_processor, generation_args


def load_maira_model(path, use_fp16=True, eval_mode=True):
    torch_dtype = torch.float16 if use_fp16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(path,
                                                 torch_dtype=torch_dtype,
                                                 trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)

    if eval_mode:
        model = model.eval()

    print("Maira model loaded")
    return model, processor


def load_radio_bench():
    print("Loading chexpert dataset...")
    radiology_bench = load_dataset("ghazal-zamani/radio_benchmark")['validation']
    # ch_plus = radiology_bench.filter(lambda x: x['dataset_name'] == 'chexpert_plus')
    ch_plus = radiology_bench
    print(f"Number of validation samples: {len(ch_plus)}")
    return ch_plus


def load_test_bench():
    print("Loading dataset")
    # radiology_bench = load_dataset("/mnt/disk2/ghazal.zamaninezhad/data/test_radio")
    radiology_bench = load_dataset(
        "parquet",
        data_files={
            # "validation": "/mnt/disk2/ghazal.zamaninezhad/data/test_radio/data/validation-*.parquet",
            "test": "/mnt/disk2/ghazal.zamaninezhad/data/test_radio/data/test-*.parquet"
        },
        cache_dir="/mnt/disk2/ghazal.zamaninezhad/hf_cache"
    )
    # print(f"Number of validation samples: {len(radiology_bench['validation'])}")
    print(f"Number of test samples: {len(radiology_bench['test'])}")
    return radiology_bench

def xray_transform(img, transform):
    # Prepare the image:
    img = np.array(img)
    img = xrv.datasets.normalize(img, 255)  # convert 8-bit image to [-1024, 1024] range
    # TODO ? handle RGB images (as it is in their repo)
    img = img[None, ...]

    img = transform(img)
    img = torch.from_numpy(img)
    return img


def extract_sections(report_text):
    findings = ""
    impression = ""

    # Normalize case
    # text = report_text.upper()

    # Pattern for FINDINGS
    findings_match = re.search(r"FINDINGS:\s*(.*?)(IMPRESSION:|SUMMARY|END OF IMPRESSION|INDICATION:|TECHNIQUE:|ACCESSION NUMBER|BY:|$)", report_text, re.DOTALL)
    if findings_match:
        findings = findings_match.group(1).strip()

    # Pattern for IMPRESSION
    impression_match = re.search(r"IMPRESSION:\s*(.*?)(END OF IMPRESSION|SUMMARY|BY:|$)", report_text, re.DOTALL)
    if impression_match:
        impression = impression_match.group(1).strip()

    return findings, impression


def save_list_to_txt(my_list, file_path):
    with open(file_path, "w") as f:
        for item in my_list:
            f.write(f"{item}\n")


def save_list_to_jsonl(my_list, file_path):
    with open(file_path, 'w') as f:
        for item in my_list:
            # assume my_list is a list of dictionaries
            if not item.values():
                continue
            f.write(json.dumps(item) + '\n')


def read_jsonl_as_dict(file_path):
    out_dict = {}
    with open(file_path, 'r') as f:
        for line in f:
            json_dict = json.loads(line)
            out_dict.update(json_dict)
    return out_dict