from transformers import BertTokenizer, ViTImageProcessor, VisionEncoderDecoderModel


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


def save_list_to_text(my_list, file_path):
    with open(file_path, "w") as f:
        for item in my_list:
            f.write(f"{item}\n")