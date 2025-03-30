import os
# set hf cache
os.environ['HF_HOME'] = "/home/m_nobakhtian/mmed/hf_cache"
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


    refs_findings, refs_impression, hyps = generate_predictions_in_batches(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        # hf_dataset=ch_plus.select(range(10)),
        hf_dataset=ch_plus,
        generation_args=generation_args,
        batch_size=32,
    )
    # print(hyps)
    output_path = "/home/m_nobakhtian/mmed/Radio-RAG/output"
    # name format: dataset_i/f_model_ref/pred
    save_list_to_text(hyps, os.path.join(output_path,
                                         'chplus_i_chexpert-mimic-cxr-impression_pred.txt'))

if __name__ == "__main__":
    main()