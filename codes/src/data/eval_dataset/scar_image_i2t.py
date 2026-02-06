import os
import sys

from datasets import load_dataset
from src.data.eval_dataset.base_eval_dataset import AutoEvalPairDataset, add_metainfo_hook, RESOLUTION_MAPPING
from src.model.processor import process_input_text


@add_metainfo_hook
def data_prepare(batch_dict, *args, **kwargs):
    image_resolution, model_backbone = kwargs['image_resolution'], kwargs['model_backbone']
    image_root = kwargs['image_root']

    query_texts, query_images, cand_texts, cand_images, dataset_infos, bboxes = [], [], [], [], [], []
    for qry_text, qry_img_path, tgt_texts, bbox in (
            zip(batch_dict['qry_text'], batch_dict['qry_image_path'], batch_dict['tgt_text'], batch_dict['bbox'])):
        
        bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
        qry_text = qry_text.replace("<|image_1|>", "")      # [NOTE]: Remove the default image token
        qry_text = process_input_text("", model_backbone, text=qry_text, add_image_token=True)
        # to stay consistent with v1 eval
        qry_text = qry_text.replace(" \n", "\n") + "\n"
        qry_text = f"{qry_text}Referring object bbox: {bbox}"
        query_texts.append([qry_text])
        qry_img_path = os.path.join(image_root, qry_img_path)
        query_images.append([{"bytes": [None], "paths": [qry_img_path],
                            "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)]}])
        
        bboxes.append(bbox)
        cand_texts.append(tgt_texts)
        cand_images.append([None] * len(tgt_texts))
        dataset_infos.append({
            "cand_names": tgt_texts,
            "label_name": tgt_texts[0],
        })

    return {"query_text": query_texts, "query_image": query_images,
            "cand_text": cand_texts, "cand_image": cand_images,
            "dataset_infos": dataset_infos, "bboxes": bboxes}


DATASET_PARSER_NAME = "scar_i2t"
DATASET_PATH = "../../data/SCaR-eval"
@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_scar_i2t_dataset(model_args, data_args, *args, **kwargs):
    dataset_name = kwargs["dataset_name"]

    # dataset = load_dataset(DATASET_HF_PATH, dataset_name, split="test")

    dataset_dict = load_dataset("parquet", data_files=f"{DATASET_PATH}/{dataset_name}.parquet")
    dataset = dataset_dict["train"]  # [NOTE]: Tentatively set to train, but actually it is test
    
    num_sample_per_subset = kwargs.get("num_sample_per_subset", sys.maxsize)
    if num_sample_per_subset is not None and type(num_sample_per_subset) is str and num_sample_per_subset.isdigit():
        num_sample_per_subset = int(num_sample_per_subset)
    if num_sample_per_subset < dataset.num_rows:
        dataset = dataset.select(range(num_sample_per_subset))
        print(f"Subsample to {len(dataset)} samples")

    kwargs['model_backbone'] = model_args.model_backbone
    kwargs['image_resolution'] = data_args.image_resolution

    dataset = dataset.map(lambda x: data_prepare(x, **kwargs), batched=True,
                          batch_size=256, num_proc=4,
                          drop_last_batch=False, load_from_cache_file=False)
    dataset = dataset.select_columns(["query_text", "query_image", "cand_text", "cand_image", "dataset_infos", "bboxes"])

    return dataset, None
