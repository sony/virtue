import os
import sys

DEMO_METADATA_DIR = "./demo_metadata"
os.makedirs(DEMO_METADATA_DIR, exist_ok=True)
temp_dir = os.path.join(DEMO_METADATA_DIR, "temp")
os.makedirs(temp_dir, exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = temp_dir

import gradio as gr
import torch
import numpy as np
import pickle
from PIL import Image, ImageDraw
from typing import Optional, Tuple, List
import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import HfArgumentParser

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.arguments import ModelArguments, DataArguments, TrainingArguments
from src.model.model import MMEBModel
from src.model.processor import load_processor, get_backbone_name

# Paths
dataset_name = "wild"
MSCOCO_IMAGES_DIR = f"./demo_metadata/{dataset_name}"
EMBEDDINGS_PATH = os.path.join(DEMO_METADATA_DIR, f"{dataset_name}_embeddings_7B_highres.pkl")
IMAGE_PATHS_PATH = os.path.join(DEMO_METADATA_DIR, f"{dataset_name}_image_paths_7B_highres.txt")

# Global variables
model = None
processor = None
candidate_embeddings = None
candidate_image_paths = []

def load_model(config: DictConfig):
    """Load VIRTUE model"""
    global model, processor
    
    print("Loading VIRTUE model...")
    
    # Set deterministic behavior
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(42)
    
    config = OmegaConf.to_container(config, resolve=True)
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_dict(config)
    
    processor = load_processor(model_args, data_args)
    model = MMEBModel.load(model_args, is_trainable=False, processor=processor)
    model.eval()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device, dtype=torch.bfloat16)
    
    print("Model loaded successfully!")

def get_image_paths():
    """Get image paths from MSCOCO directory"""
    image_paths = []
    for root, dirs, files in os.walk(MSCOCO_IMAGES_DIR):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, MSCOCO_IMAGES_DIR)
                image_paths.append(rel_path)
    return sorted(image_paths)

def encode_images():
    """Encode all images and save embeddings (with batch processing)"""
    global model, processor
    
    print("Encoding images...")
    
    image_paths = get_image_paths()
    print(f"Found {len(image_paths)} images")
    
    # Get model backbone and image token
    from src.model.processor import VLM_IMAGE_TOKENS, get_backbone_name
    from transformers import AutoConfig
    
    config = AutoConfig.from_pretrained(model.config._name_or_path, trust_remote_code=True)
    model_backbone = get_backbone_name(hf_config=config)
    image_token = VLM_IMAGE_TOKENS[model_backbone]
    
    # Batch processing parameters
    batch_size = 8
    embedding_dict = {}
    valid_paths = []
    
    # Process images in batches
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(image_paths) + batch_size - 1)//batch_size}")
        
        # Load batch images
        batch_images = []
        batch_valid_paths = []
        for img_path in batch_paths:
            try:
                full_path = os.path.join(MSCOCO_IMAGES_DIR, img_path)
                image = Image.open(full_path).convert('RGB')
                batch_images.append(image)
                batch_valid_paths.append(img_path)
            except Exception as e:
                print(f"Failed to load {img_path}: {e}")
                continue
        
        if not batch_images:
            continue
        
        # Prepare batch inputs
        model_inputs = {
            'text': [f"{image_token}\nRepresent the given image.\n"] * len(batch_images),
            'images': batch_images
        }
        
        # Process and encode batch
        from src.model.processor import process_vlm_inputs_fns
        process_fn = process_vlm_inputs_fns[model_backbone]
        inputs = process_fn(model_inputs, processor=processor, max_length=512)
        
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        
        with torch.no_grad():
            with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                output = model(qry=inputs)
                batch_embeddings = output["qry_reps"].detach().cpu().numpy()
        
        # Store batch embeddings
        for j, img_path in enumerate(batch_valid_paths):
            if batch_embeddings.ndim == 2:
                embedding_dict[img_path] = batch_embeddings[j]
            else:
                embedding_dict[img_path] = batch_embeddings
        
        valid_paths.extend(batch_valid_paths)
    
    # Save
    with open(EMBEDDINGS_PATH, 'wb') as f:
        pickle.dump(embedding_dict, f)
    
    with open(IMAGE_PATHS_PATH, 'w') as f:
        for img_path in valid_paths:
            f.write(f"{img_path}\n")
    
    print(f"Saved {len(embedding_dict)} embeddings")
    return embedding_dict, valid_paths

def load_embeddings():
    """Load precomputed embeddings"""
    global candidate_embeddings, candidate_image_paths
    
    print("Loading embeddings...")
    
    if os.path.exists(EMBEDDINGS_PATH) and os.path.exists(IMAGE_PATHS_PATH):
        with open(EMBEDDINGS_PATH, 'rb') as f:
            candidate_embeddings = pickle.load(f)
        
        with open(IMAGE_PATHS_PATH, 'r') as f:
            candidate_image_paths = [line.strip() for line in f.readlines()]
    else:
        candidate_embeddings, candidate_image_paths = encode_images()
    
    print(f"Ready with {len(candidate_embeddings)} embeddings")

def encode_query(image: Image.Image, text: str = None, point: Tuple[float, float] = None, output_type: str = "images", bbox: Tuple[float, float, float, float] = None) -> np.ndarray:
    """Encode a query"""
    global model, processor
    
    # Ensure deterministic behavior for each forward pass
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Get model backbone and image token
    from src.model.processor import VLM_IMAGE_TOKENS, get_backbone_name
    from transformers import AutoConfig
    
    config = AutoConfig.from_pretrained(model.config._name_or_path, trust_remote_code=True)
    model_backbone = get_backbone_name(hf_config=config)
    image_token = VLM_IMAGE_TOKENS[model_backbone]
    
    # Prepare text
    if text:
        if output_type == "images":
            query_text = f"{image_token}\nRetrieve an image that matches: {text}\n"
        else:
            query_text = f"{image_token}\nRetrieve a caption that describes: {text}\n"
    elif point:
        x, y = point
        img_width, img_height = image.size
        pixel_x = int(x * img_width)
        pixel_y = int(y * img_height)
        
        if output_type == "images":
            query_text = f"{image_token}\nFind the caption that best describes the segmented object, considering both local details and global context in the given image.\nReferring object point: ({pixel_x}, {pixel_y})"
        else:
            query_text = f"{image_token}\nFind the caption that best describes the segmented object, considering both local details and global context in the given image.\nReferring object point: ({pixel_x}, {pixel_y})"
    elif bbox:
        x1, y1, x2, y2 = bbox
        img_width, img_height = image.size
        pixel_x1 = int(x1 * img_width)
        pixel_y1 = int(y1 * img_height)
        pixel_x2 = int(x2 * img_width)
        pixel_y2 = int(y2 * img_height)
        
        if output_type == "images":
            query_text = f"{image_token}\nFind the caption that best describes the object in the bounding box, considering both local details and global context in the given image.\nReferring object bbox: ({pixel_x1}, {pixel_y1}, {pixel_x2}, {pixel_y2})"
        else:
            query_text = f"{image_token}\nFind the caption that best describes the object in the bounding box, considering both local details and global context in the given image.\nReferring object bbox: ({pixel_x1}, {pixel_y1}, {pixel_x2}, {pixel_y2})"
    else:
        if output_type == "images":
            query_text = f"{image_token}\nRetrieve an image similar to the given image.\n"
        else:
            query_text = f"{image_token}\nRetrieve a caption that describes the given image.\n"
    
    print(query_text)

    # Process and encode
    model_inputs = {
        'text': [query_text],
        'images': [image]
    }
    
    from src.model.processor import process_vlm_inputs_fns
    process_fn = process_vlm_inputs_fns[model_backbone]
    inputs = process_fn(model_inputs, processor=processor)  # max_length=512

    if point is not None:
        # Convert normalized coordinates [0,1] to SAM2's (1024, 1024) coordinate system
        point_x, point_y = point
        # SAM2Transforms converts images to (1024, 1024), so we need to scale the point accordingly
        sam_size = 1024
        sam_x = point_x * sam_size
        sam_y = point_y * sam_size
        inputs['point'] = [sam_x, sam_y]
    elif bbox is not None:
        # Convert normalized bbox coordinates [0,1] to SAM2's (1024, 1024) coordinate system
        x1, y1, x2, y2 = bbox
        sam_size = 1024
        sam_x1 = x1 * sam_size
        sam_y1 = y1 * sam_size
        sam_x2 = x2 * sam_size
        sam_y2 = y2 * sam_size
        inputs['bbox'] = [sam_x1, sam_y1, sam_x2, sam_y2]
    else:
        inputs['point'] = None
        inputs['bbox'] = None
    
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    
    with torch.no_grad():
        with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
            output = model(qry=inputs)
            query_embedding = output["qry_reps"].detach().cpu().numpy()
    
    if query_embedding.ndim == 1:
        return query_embedding
    else:
        return query_embedding[0]

def crop_image_by_bbox(image: Image.Image, bbox: Tuple[float, float, float, float]) -> Image.Image:
    """Crop image based on normalized bounding box coordinates"""
    x1, y1, x2, y2 = bbox
    img_width, img_height = image.size
    
    # Convert normalized coordinates to pixel coordinates
    pixel_x1 = int(x1 * img_width)
    pixel_y1 = int(y1 * img_height)
    pixel_x2 = int(x2 * img_width)
    pixel_y2 = int(y2 * img_height)
    
    # Ensure coordinates are within image bounds
    pixel_x1 = max(0, min(pixel_x1, img_width))
    pixel_y1 = max(0, min(pixel_y1, img_height))
    pixel_x2 = max(0, min(pixel_x2, img_width))
    pixel_y2 = max(0, min(pixel_y2, img_height))
    
    # Ensure x2 > x1 and y2 > y1
    if pixel_x2 <= pixel_x1:
        pixel_x2 = pixel_x1 + 1
    if pixel_y2 <= pixel_y1:
        pixel_y2 = pixel_y1 + 1
    
    # Crop the image
    cropped = image.crop((pixel_x1, pixel_y1, pixel_x2, pixel_y2))
    return cropped

def compute_similarities(query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
    """Compute similarities between query and candidates"""
    global candidate_embeddings, candidate_image_paths
    
    similarities = []
    
    for img_path in candidate_image_paths:
        cand_embedding = candidate_embeddings[img_path]
        
        if torch.is_tensor(cand_embedding):
            cand_embedding = cand_embedding.detach().cpu().numpy()
        if torch.is_tensor(query_embedding):
            query_embedding = query_embedding.detach().cpu().numpy()
        
        if cand_embedding.ndim > 1:
            cand_embedding = cand_embedding.flatten()
        if query_embedding.ndim > 1:
            query_embedding = query_embedding.flatten()
        
        similarity = np.dot(query_embedding, cand_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(cand_embedding)
        )
        
        similarities.append((img_path, similarity))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

def retrieve_results(image: Image.Image, text: str = None, point: Tuple[float, float] = None, 
                    output_type: str = "images", top_k: int = 5, bbox: Tuple[float, float, float, float] = None) -> Tuple[List, str]:
    """Main retrieval function"""
    query_embedding = encode_query(image, text, point, output_type, bbox)
    similarities = compute_similarities(query_embedding, top_k)
    
    if output_type == "images":
        gallery_results = []
        for img_path, similarity in similarities:
            try:
                full_path = os.path.join(MSCOCO_IMAGES_DIR, img_path)
                pil_image = Image.open(full_path).convert('RGB')
                gallery_results.append((pil_image, f"Similarity: {similarity:.3f}\nImage: {img_path}"))
            except Exception as e:
                print(f"Failed to load {img_path}: {e}")
                continue
        return gallery_results, ""
    else:
        text_results = []
        for img_path, similarity in similarities:
            text_results.append(f"Image: {img_path}\nSimilarity: {similarity:.3f}")
        return [], "\n\n---\n\n".join(text_results)

def demo_interface(image, text, point_x, point_y, bbox_x1, bbox_y1, bbox_x2, bbox_y2, output_type, top_k, input_mode):
    """Gradio interface function"""
    if image is None:
        return [], "Please upload an image.", None
    
    point = None
    bbox = None
    cropped_image = None
    
    if input_mode == "point":
        if point_x is not None and point_y is not None:
            point = (float(point_x), float(point_y))
    elif input_mode == "bbox":
        if bbox_x1 is not None and bbox_y1 is not None and bbox_x2 is not None and bbox_y2 is not None:
            bbox = (float(bbox_x1), float(bbox_y1), float(bbox_x2), float(bbox_y2))
    elif input_mode == "crop":
        if bbox_x1 is not None and bbox_y1 is not None and bbox_x2 is not None and bbox_y2 is not None:
            bbox = (float(bbox_x1), float(bbox_y1), float(bbox_x2), float(bbox_y2))
            # Crop the image
            cropped_image = crop_image_by_bbox(image, bbox)
    
    # Use cropped image for processing if available, otherwise use original
    processing_image = cropped_image if cropped_image is not None else image
    
    gallery_results, text_results = retrieve_results(processing_image, text, point, output_type, int(top_k), bbox)
    
    # Return appropriate image for display
    display_image = cropped_image if input_mode == "crop" else image
    
    return gallery_results, text_results, display_image

@hydra.main(version_base=None, config_path="./configs", config_name="virtue_demo")
def main(config: DictConfig):
    # Load model and embeddings
    print("Initializing VIRTUE Demo...")
    load_model(config)
    load_embeddings()

    # Create Gradio interface
    with gr.Blocks(title="VIRTUE Demo") as demo:
        gr.Markdown("# VIRTUE Demo")
        gr.Markdown("Upload an image and optionally provide text, click, or bounding box on the image to set a point or bounding box for finding similar images.")
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    type="pil", 
                    interactive=True,
                    label="Upload Image (click to set a point)"
                )
                
                with gr.Row():
                    text_input = gr.Textbox(label="Text Query (optional)", placeholder="Describe what you're looking for...")
                
                with gr.Row():
                    point_x = gr.Number(label="Point X (normalized)", visible=False)
                    point_y = gr.Number(label="Point Y (normalized)", visible=False)
                
                with gr.Row():
                    bbox_x1 = gr.Number(label="BBox X1 (normalized)", visible=False)
                    bbox_y1 = gr.Number(label="BBox Y1 (normalized)", visible=False)
                    bbox_x2 = gr.Number(label="BBox X2 (normalized)", visible=False)
                    bbox_y2 = gr.Number(label="BBox Y2 (normalized)", visible=False)
                
                with gr.Row():
                    output_type = gr.Radio(
                        choices=["images", "text"], 
                        value="images", 
                        label="Output Type"
                    )
                    top_k = gr.Slider(minimum=1, maximum=100, step=1, value=1, label="Top K Results")
                
                with gr.Row():
                    input_mode = gr.Radio(
                        choices=["point", "bbox", "crop"], 
                        value="point", 
                        label="Input Mode"
                    )
                
                submit_btn = gr.Button("Search", variant="primary")
                clear_btn = gr.Button("Clear Selection", variant="secondary")
            
            with gr.Column():
                output_gallery = gr.Gallery(
                    label="Retrieved Images",
                    show_label=True,
                    elem_id="gallery",
                    columns=2,
                    rows=3,
                    height="auto"
                )
                output_text = gr.Textbox(
                    label="Retrieved Results",
                    lines=10,
                    max_lines=20
                )
        
        def update_output_visibility(output_type_val):
            if output_type_val == "images":
                return gr.Gallery(visible=True), gr.Textbox(visible=False)
            else:
                return gr.Gallery(visible=False), gr.Textbox(visible=True)
        
        # Store original image and state for clearing
        original_image = None
        click_state = {"mode": "point", "first_click": None, "second_click": None}
        
        def handle_image_click_with_storage(image, evt: gr.SelectData, input_mode):
            """Handle image click events and store original image"""
            nonlocal original_image, click_state
            if image is None:
                return None, None, None, None, None, None, None
            
            # Store original image on first click
            if original_image is None:
                original_image = image.copy()
            
            x = evt.index[0]
            y = evt.index[1]
            
            img_width, img_height = image.size
            norm_x = x / img_width
            norm_y = y / img_height
            
            img_with_annotation = image.copy()
            draw = ImageDraw.Draw(img_with_annotation)
            
            if input_mode == "point":
                # Point mode - single click
                r = max(10, min(img_width, img_height) // 50)
                draw.ellipse((x - r, y - r, x + r, y + r), fill="red", outline="white", width=2)
                return img_with_annotation, norm_x, norm_y, None, None, None, None
                
            elif input_mode in ["bbox", "crop"]:
                # Bbox/Crop mode - two clicks
                if click_state["first_click"] is None:
                    # First click - store coordinates
                    click_state["first_click"] = (norm_x, norm_y)
                    r = max(10, min(img_width, img_height) // 50)
                    draw.ellipse((x - r, y - r, x + r, y + r), fill="blue", outline="white", width=2)
                    return img_with_annotation, None, None, norm_x, norm_y, None, None
                else:
                    # Second click - create bbox
                    first_x, first_y = click_state["first_click"]
                    click_state["second_click"] = (norm_x, norm_y)
                    
                    # Draw first point
                    r = max(10, min(img_width, img_height) // 50)
                    first_pixel_x = int(first_x * img_width)
                    first_pixel_y = int(first_y * img_height)
                    draw.ellipse((first_pixel_x - r, first_pixel_y - r, first_pixel_x + r, first_pixel_y + r), 
                               fill="blue", outline="white", width=2)
                    
                    # Draw second point
                    second_pixel_x = int(norm_x * img_width)
                    second_pixel_y = int(norm_y * img_height)
                    draw.ellipse((second_pixel_x - r, second_pixel_y - r, second_pixel_x + r, second_pixel_y + r), 
                               fill="red", outline="white", width=2)
                    
                    # Draw bounding box rectangle
                    min_x = min(first_pixel_x, second_pixel_x)
                    min_y = min(first_pixel_y, second_pixel_y)
                    max_x = max(first_pixel_x, second_pixel_x)
                    max_y = max(first_pixel_y, second_pixel_y)
                    draw.rectangle([min_x, min_y, max_x, max_y], outline="red", width=4)
                    
                    # Reset state for next bbox
                    click_state["first_click"] = None
                    click_state["second_click"] = None
                    
                    return img_with_annotation, None, None, first_x, first_y, norm_x, norm_y
        
        def clear_selection(image):
            nonlocal original_image, click_state
            if original_image is None:
                return image, None, None, None, None, None, None
            # Reset click state
            click_state["first_click"] = None
            click_state["second_click"] = None
            return original_image, None, None, None, None, None, None
        
        # Event handlers
        image_input.select(
            handle_image_click_with_storage,
            inputs=[image_input, input_mode],
            outputs=[image_input, point_x, point_y, bbox_x1, bbox_y1, bbox_x2, bbox_y2]
        )
        
        clear_btn.click(
            clear_selection,
            inputs=[image_input],
            outputs=[image_input, point_x, point_y, bbox_x1, bbox_y1, bbox_x2, bbox_y2]
        )
        
        output_type.change(
            update_output_visibility,
            inputs=[output_type],
            outputs=[output_gallery, output_text]
        )
        
        submit_btn.click(
            demo_interface,
            inputs=[image_input, text_input, point_x, point_y, bbox_x1, bbox_y1, bbox_x2, bbox_y2, output_type, top_k, input_mode],
            outputs=[output_gallery, output_text, image_input]
        )

    # Launch the demo
    demo.launch(
        share=False, 
        server_name="0.0.0.0", 
        server_port=7864, 
        allowed_paths=[DEMO_METADATA_DIR]
    )

if __name__ == "__main__":
    import traceback
    from hydra.core.global_hydra import GlobalHydra

    if GlobalHydra.instance().is_initialized():
        print("--- Hydra was already initialized by: ---")
        traceback.print_stack()
        GlobalHydra.instance().clear()

    main()
