import transformers.trainer as _tr; _tr.check_torch_load_is_safe = lambda: None
import transformers.modeling_utils as _mu; _mu.check_torch_load_is_safe = lambda: None
####### BREAK TORCH LOAD ISSUE < 2.6 disable the CVE‐enforced check #######


import os
import sys
import torch
import numpy as np
import json
import hydra
from hydra.core.global_hydra import GlobalHydra
from PIL import Image

# Add parent directory to path for src imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.arguments import ModelArguments, DataArguments, TrainingArguments
from src.model.model import MMEBModel
from src.model.processor import load_processor, VLM_IMAGE_TOKENS, get_backbone_name, process_vlm_inputs_fns
from transformers import AutoConfig


# Initialize Hydra for SAM2 loading
if not GlobalHydra().is_initialized():
    hydra.initialize(config_path="./configs", version_base=None)

# Determinism
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(42)

model_dir = 'Sony/VIRTUE-2B-SCaR'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True, token=True)

# Build arguments directly (no YAML required)
model_args = ModelArguments(
    model_name=model_dir,
    checkpoint_path=None,
    pooling="last",
    normalize=True,
    lora=False,
    model_backbone='qwen2_vl',
)
persisted_sam = config.virtue_sam

model_args.sam = True
model_args.sam_config = {
    "config_path": persisted_sam.get('config_path') if persisted_sam else None,
    "checkpoint": persisted_sam.get('checkpoint') if persisted_sam else None,
    "points_per_side": (persisted_sam.get('points_per_side') if persisted_sam else 16),
    "feature_levels": (persisted_sam.get('feature_levels') if persisted_sam else 3),
}

data_args = DataArguments()
training_args = TrainingArguments()

processor = load_processor(model_args, data_args)
model = MMEBModel.load(model_args, is_trainable=False, processor=processor)
model.eval()
model = model.to(device, dtype=torch.bfloat16)

# Get model backbone and image token
model_backbone = get_backbone_name(hf_config=config)
image_token = VLM_IMAGE_TOKENS[model_backbone]

# Image + Text -> Text
image_path = '../assets/example.jpg'
image = Image.open(image_path).convert('RGB')

model_inputs = {
    'text': [f"{image_token}\nRepresent the given image with the following question: What is in the image"],
    'images': [image]
}

process_fn = process_vlm_inputs_fns[model_backbone]
inputs = process_fn(model_inputs, processor=processor, max_length=512)
device = next(model.parameters()).device
inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

with torch.no_grad():
    with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
        qry_output = model(qry=inputs)["qry_reps"]

# Candidates for all scenarios
test_strings = ['A cat', 'A dog', 'A tiger']

# Scenario 1: No visual prompts (image only)
print("\n--- Similarities (no visual prompts) ---")
for string in test_strings:
    cand_inputs = process_fn({'text': [string], 'images': [None]}, processor=processor)
    cand_inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in cand_inputs.items()}
    with torch.no_grad():
        with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
            tgt_output = model(tgt=cand_inputs)["tgt_reps"]
    sim = model.compute_similarity(qry_output, tgt_output)
    print(f"no-prompt | {string} = {sim}")

'''
--- Similarities (no visual prompts) ---
no-prompt | A cat = tensor([[0.3030]], device='cuda:0')
no-prompt | A dog = tensor([[0.2453]], device='cuda:0')
no-prompt | A tiger = tensor([[0.1714]], device='cuda:0')
'''

# Scenario 2: Point prompts — two examples (left/right)
print("\n--- Similarities (point prompts) ---")
sam_size = 1024  # SAM2Transforms output size
point_examples = [(0.25, 0.5), (0.75, 0.5)]
for (px, py) in point_examples:
    point_text = f"{image_token}\nFind the caption that best describes the segmented object, considering both local details and global context in the given image.\nReferring object point: ({int(px*image.size[0])}, {int(py*image.size[1])})"
    q_inputs = process_fn({'text': [point_text], 'images': [image]}, processor=processor)
    q_inputs['point'] = [px * sam_size, py * sam_size]
    q_inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in q_inputs.items()}
    with torch.no_grad():
        with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
            point_qry = model(qry=q_inputs)["qry_reps"]
    for string in test_strings:
        cand_inputs = process_fn({'text': [string], 'images': [None]}, processor=processor)
        cand_inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in cand_inputs.items()}
        with torch.no_grad():
            with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                tgt_output = model(tgt=cand_inputs)["tgt_reps"]
        sim = model.compute_similarity(point_qry, tgt_output)
        print(f"point ({px:.2f},{py:.2f}) | {string} = {sim}")

'''
--- Similarities (point prompts) ---
point (0.25,0.50) | A cat = tensor([[0.1793]], device='cuda:0')
point (0.25,0.50) | A dog = tensor([[0.1339]], device='cuda:0')
point (0.25,0.50) | A tiger = tensor([[0.1314]], device='cuda:0')
point (0.75,0.50) | A cat = tensor([[0.2232]], device='cuda:0')
point (0.75,0.50) | A dog = tensor([[0.1742]], device='cuda:0')
point (0.75,0.50) | A tiger = tensor([[0.1692]], device='cuda:0')
'''

# Scenario 3: BBox prompts — two examples (left/right)
print("\n--- Similarities (bbox prompts) ---")
bbox_examples = [
    (0.05, 0.20, 0.45, 0.80),  # left
    (0.55, 0.20, 0.95, 0.80),  # right
]
for (x1, y1, x2, y2) in bbox_examples:
    bbox_text = f"{image_token}\nFind the caption that best describes the object in the bounding box, considering both local details and global context in the given image.\nReferring object bbox: ({int(x1*image.size[0])}, {int(y1*image.size[1])}, {int(x2*image.size[0])}, {int(y2*image.size[1])})"
    q_inputs = process_fn({'text': [bbox_text], 'images': [image]}, processor=processor)
    q_inputs['bbox'] = [x1 * sam_size, y1 * sam_size, x2 * sam_size, y2 * sam_size]
    q_inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in q_inputs.items()}
    with torch.no_grad():
        with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
            bbox_qry = model(qry=q_inputs)["qry_reps"]
    for string in test_strings:
        cand_inputs = process_fn({'text': [string], 'images': [None]}, processor=processor)
        cand_inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in cand_inputs.items()}
        with torch.no_grad():
            with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                tgt_output = model(tgt=cand_inputs)["tgt_reps"]
        sim = model.compute_similarity(bbox_qry, tgt_output)
        print(f"bbox ({x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f}) | {string} = {sim}")

'''
--- Similarities (bbox prompts) ---
bbox (0.05,0.20,0.45,0.80) | A cat = tensor([[0.2100]], device='cuda:0')
bbox (0.05,0.20,0.45,0.80) | A dog = tensor([[0.1512]], device='cuda:0')
bbox (0.05,0.20,0.45,0.80) | A tiger = tensor([[0.1719]], device='cuda:0')
bbox (0.55,0.20,0.95,0.80) | A cat = tensor([[0.1583]], device='cuda:0')
bbox (0.55,0.20,0.95,0.80) | A dog = tensor([[0.1953]], device='cuda:0')
bbox (0.55,0.20,0.95,0.80) | A tiger = tensor([[0.1225]], device='cuda:0')
'''