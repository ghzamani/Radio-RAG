import os
# set hf cache
os.environ['HF_HOME'] = "/home/m_nobakhtian/mmed/hf_cache"

import pandas as pd
from datasets import load_dataset
from utills import *
from inference import generate_predictions_in_batches


def main():

    # Load chexpert dataset
    print("Loading chexpert dataset...")
    radiology_bench = load_dataset("MiladMola/radio_benchmark")['validation']
    ch_plus = radiology_bench.filter(lambda x:x['dataset_name']=='chexpert_plus')
    print(f"Number of validation samples: {len(ch_plus)}")

    # Load model
    # TODO support all models
    print("Loading the model...")
    model, tokenizer, image_processor, generation_args = load_chexpert_model(
        "IAMJB/chexpert-mimic-cxr-impression-baseline"
    )


    refs, hyps = generate_predictions_in_batches(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        hf_dataset=ch_plus.select(range(10)),
        generation_args=generation_args,
        batch_size=4,
    )
    print(hyps)

if __name__ == "__main__":
    main()