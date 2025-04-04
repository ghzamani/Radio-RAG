import os
# set hf cache
os.environ['HF_HOME'] = "/home/m_nobakhtian/mmed/hf_cache"

from utills import *
from inference import chexpert_generate_predictions_in_batches, chexpert_generate_predictions


def main():
    ch_plus = load_radio_bench()

    # Load model
    # TODO support all models
    print("Loading the model...")
    model, tokenizer, image_processor, generation_args = load_chexpert_model(
        "IAMJB/chexpert-findings-baseline"
    )

    # check whether batch prediction is working as expected .... DONE
    refs_findings, refs_impression, hyps = chexpert_generate_predictions_in_batches(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        # hf_dataset=ch_plus.select(range(7)),
        hf_dataset=ch_plus,
        generation_args=generation_args,
        batch_size=32,
    )
    # print(refs_findings)
    # print(hyps)
    output_path = "/home/m_nobakhtian/mmed/Radio-RAG/outputs"
    # name format: {dataset}_{impression/findings}_model_{ref/pred}
    save_list_to_txt(hyps, os.path.join(output_path,
                                         'chplus_findings_chexpert-mimic-cxr-impression_pred.txt'))
    save_list_to_txt(refs_findings, os.path.join(output_path,
                                        'chplus_findings_chexpert-mimic-cxr-impression_ref.txt'))

if __name__ == "__main__":
    main()