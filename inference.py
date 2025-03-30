import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GenerationConfig
from radiology_dataset import RadiologyDataset, RadiologyCollator


def generate_predictions_in_batches(
    model,
    tokenizer,
    image_processor,
    hf_dataset,  # Your original HF dataset remains unchanged
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

