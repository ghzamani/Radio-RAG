from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoProcessor,
                          BertTokenizer, ViTImageProcessor,
                          VisionEncoderDecoderModel)
import torch


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


def save_list_to_txt(my_list, file_path):
    with open(file_path, "w") as f:
        for item in my_list:
            f.write(f"{item}\n")