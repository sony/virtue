import torch
import numpy as np
import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf

# Add parent directory to path for src imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.arguments import ModelArguments, DataArguments, TrainingArguments
from src.model.model import MMEBModel
from src.model.processor import load_processor
from transformers import HfArgumentParser


@hydra.main(version_base=None, config_path="../configs", config_name="virtue_eval")
def main(config: DictConfig):
    # Set deterministic behavior
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(42)
    
    # Load args from yaml config
    cfg_dict = OmegaConf.to_container(config, resolve=True)
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_dict(cfg_dict)
    
    processor = load_processor(model_args, data_args)
    model = MMEBModel.load(model_args, is_trainable=False, processor=processor)
    model.eval()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device, dtype=torch.bfloat16)

    output_dir = '../VIRTUE-7B-FT-SCAR/'
    os.makedirs(output_dir, exist_ok=True)

    model.encoder._hf_peft_config_loaded = False
    model.encoder.save_pretrained(output_dir, safe_serialization=False)
    
    # Ensure SAM reducer and projection weights are available in the HF folder
    # Prefer copying from the training checkpoint if present; otherwise save from the in-memory model
    sam_ckpt_src = os.path.join(model_args.checkpoint_path, 'sam2_reducer_and_vlm_layer.pth')
    sam_ckpt_dst = os.path.join(output_dir, 'sam2_reducer_and_vlm_layer.pth')
    if os.path.exists(sam_ckpt_src):
        import shutil
        shutil.copy2(sam_ckpt_src, sam_ckpt_dst)
    elif getattr(model, 'segmentation_model', None) is not None:
        torch.save({
            'sam2_reducer': model.segmentation_model.sam2_reducer.state_dict(),
            'sam2vlm_layer': model.segmentation_model.sam2vlm_layer.state_dict()
        }, sam_ckpt_dst)

    # Persist SAM-related configs into config.json for easier HF loading later
    cfg_path = os.path.join(output_dir, 'config.json')
    if os.path.exists(cfg_path):
        import json
        with open(cfg_path, 'r') as f:
            cfg = json.load(f)
        sam_cfg = model_args.sam_config if getattr(model_args, 'sam', False) else {}
        cfg['virtue_sam'] = {
            'enabled': bool(getattr(model_args, 'sam', False)),
            'config_path': sam_cfg.get('config_path') if sam_cfg else None,
            'checkpoint': sam_cfg.get('checkpoint') if sam_cfg else None,
            'points_per_side': sam_cfg.get('points_per_side', 16) if sam_cfg is not None else 16,
            'feature_levels': sam_cfg.get('feature_levels', 3) if sam_cfg is not None else 3
        }
        with open(cfg_path, 'w') as f:
            json.dump(cfg, f, indent=2)


if __name__ == "__main__":
    main()