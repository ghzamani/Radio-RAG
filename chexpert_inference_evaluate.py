import os
# set hf cache
os.environ['HF_HOME'] = "/home/m_nobakhtian/mmed/hf_cache"
from datasets import load_dataset

from utills import *
from inference import generate_predictions_in_batches
from evaluation import compute_scores

def evaluate_model(model, tokenizer, image_processor, dataset, generation_args,
                   mode="impression"):
    refs_findings, refs_impression, hyps = generate_predictions_in_batches(
        model,
        tokenizer,
        image_processor,
        dataset,
        generation_args=generation_args,
        batch_size=32
    )

    if mode == "impression":
        scores = compute_scores(refs_impression, hyps)
    elif mode == "findings":
        scores = compute_scores(refs_findings, hyps)
    else:
        raise ValueError("Invalid mode")
    return scores

def main():

    # Load chexpert dataset
    print("Loading chexpert dataset...")
    radiology_bench = load_dataset("MiladMola/radio_benchmark")['validation']
    ch_plus = radiology_bench.filter(lambda x: x['dataset_name'] == 'chexpert_plus')
    print(f"Number of validation samples: {len(ch_plus)}")

    # Load model
    # TODO support all models
    print("Loading the model...")
    model, tokenizer, image_processor, generation_args = load_chexpert_model(
        "IAMJB/chexpert-mimic-cxr-impression-baseline"
    )

    scores = evaluate_model(model, tokenizer, image_processor, ch_plus, generation_args, mode="impression")
    print(scores)


if __name__ == '__main__':
    main()