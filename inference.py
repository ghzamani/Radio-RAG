import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GenerationConfig
from radiology_dataset import RadiologyDataset, RadiologyCollator


def chexpert_generate_predictions_in_batches(
    model,
    tokenizer,
    image_processor,
    hf_dataset,  # original HF dataset remains unchanged
    generation_args,
    batch_size=32
):
    """
   Generate predictions in batches for a given dataset.

   Args:
       model: The HuggingFace model for generation.
       tokenizer: The tokenizer for decoding predictions.
       image_processor: The processor for image preprocessing.
       hf_dataset: Huggingface dataset.
       generation_args: Dictionary of generation config arguments.
       batch_size: Number of samples per batch.

   Returns:
       refs (list): List of reference texts.
       hyps (list): List of generated predictions.
   """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # Create dataset (converts images to tensor)
    dataset = RadiologyDataset(hf_dataset)
    # Create dataloader (applies image processor to pil images)
    custom_collate = RadiologyCollator(image_processor)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=custom_collate,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    refs_findings = []
    refs_impression = []
    hyps = []

    with torch.no_grad():
        for pixel_values_batch, findings, impressions in tqdm(dataloader, desc="Generating predictions"):
            pixel_values_batch = pixel_values_batch.to(device)
            # print(pixel_values_batch.get_device())
            # Generate predictions
            generated_ids = model.generate(
                pixel_values_batch,
                generation_config=GenerationConfig(
                    **{**generation_args,
                       "decoder_start_token_id": tokenizer.cls_token_id}
                )
            )

            generated_texts = tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )

            refs_findings.extend(findings)
            refs_impression.extend(impressions)
            hyps.extend(generated_texts)

    return refs_findings, refs_impression, hyps


def chexpert_generate_predictions(model,
                                  tokenizer,
                                  image_processor,
                                  hf_dataset,
                                  generation_args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    refs_findings = []
    refs_impression = []
    hyps = []

    with torch.no_grad():
        for data in tqdm(hf_dataset):
            image = data["image"].convert("RGB")
            pixel_values = image_processor(image, return_tensors="pt").pixel_values.to(device)
            # Generate predictions
            generated_ids = model.generate(
                pixel_values,
                generation_config=GenerationConfig(
                    **{**generation_args, "decoder_start_token_id": tokenizer.cls_token_id})
            )
            generated_texts = tokenizer.batch_decode(generated_ids,
                                                     skip_special_tokens=True)

            refs_findings.append(data["findings"])
            refs_impression.append(data["impression"])
            hyps.extend(generated_texts)

    return refs_findings, refs_impression, hyps


def maira_predict(model, processor, sample_data, device="cuda"):
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
