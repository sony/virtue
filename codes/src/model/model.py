from typing import Dict, Optional, Tuple
import os
import torch
import torch.distributed as dist
import numpy as np

from torchvision.ops.boxes import batched_nms, box_area
from torch import nn, Tensor
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoConfig
from peft import LoraConfig, get_peft_model, PeftModel

from src.arguments import ModelArguments
from src.model.processor import LLAVA_NEXT, QWEN2_VL, PHI3V, get_backbone_name, print_master, QWEN2_5_VL, \
    backbone2model, QWEN2_VL_TOKENSELECTION, QWEN2_5_VL_TOKENSELECTION
from src.arguments import ModelArguments
from src.model.processor import LLAVA_NEXT, QWEN2_VL, PHI3V, get_backbone_name, print_master, QWEN2_5_VL, INTERNVIDEO2, \
    QWEN2_VL_TOKENSELECTION, backbone2model, GME, VLM_IMAGE_TOKENS, LamRA, LamRA_QWEN2_5, COLPALI
from src.model.vlm_backbone.colpali import ColPali
from src.model.vlm_backbone.gme.gme_inference import GmeQwen2VL
from src.model.vlm_backbone.lamra.lamra_inference import LamRAQwen2VL
from src.model.vlm_backbone.lamra.lamra_qwen25_inference import LamRAQwen25VL
from src.model.vlm_backbone.phi3_v.modeling_phi3_v import Phi3VForCausalLM
from src.model.vlm_backbone.llava_next import LlavaNextForConditionalGeneration
from huggingface_hub import hf_hub_download, list_repo_files

from src.sam2.utils.amg import MaskData, generate_crop_boxes, build_all_layer_point_grids
from src.sam2.build_sam import build_sam2
from src.sam2.utils.transforms import SAM2Transforms


class SAM2VLM(nn.Module):
    def __init__(self, sam_config):
        super().__init__()

        self.sam_model = build_sam2(sam_config["config_path"], sam_config["checkpoint"], apply_postprocessing=False)
        for param in self.sam_model.parameters():
            param.requires_grad = False

        self._transforms = SAM2Transforms(
            resolution=self.sam_model.image_size,
            mask_threshold=0,
            max_hole_area=0,
            max_sprinkle_area=0,
        )

        crop_n_layers, crop_n_points_downscale_factor = 0, 1
        self.point_grids = build_all_layer_point_grids(
            sam_config["points_per_side"],
            crop_n_layers,
            crop_n_points_downscale_factor,
        )
        self.multimask_output = False
        self.num_feature_levels = sam_config["feature_levels"]

        # Spatial dim for backbone feature maps
        self._bb_feat_sizes = [
            (256, 256),
            (128, 128),
            (64, 64),
        ]

        # initialize a reducer and a projection layer
        sam_dim = self.sam_model.image_encoder.neck.convs[0].conv.weight.shape[0]                   # hardcode to find but adaptable to the model size of SAM2
        self.sam2_reducer = nn.Conv2d(in_channels=sam_dim, out_channels=sam_dim, kernel_size=4, stride=4)
        # self.sam2vlm_layer = nn.Linear(sam_dim, sam_config["vlm_hidden_size"])

        self.sam2vlm_layer = nn.Sequential(
            nn.Linear(sam_dim, sam_dim),
            nn.GELU(),
            nn.Linear(sam_dim, sam_config["vlm_hidden_size"])
        )

    def _predict(
        self,
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if point_coords is not None:
            concat_points = (point_coords, point_labels)
        else:
            concat_points = None

        if boxes is not None:
            box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=boxes.device)
            box_labels = box_labels.repeat(boxes.size(0), 1)
            # we merge "boxes" and "points" into a single "concat_points" input (where
            # boxes are added at the beginning) to sam_prompt_encoder
            if concat_points is not None:
                concat_coords = torch.cat([boxes, concat_points[0]], dim=1)
                concat_labels = torch.cat([box_labels, concat_points[1]], dim=1)
                concat_points = (concat_coords, concat_labels)
            else:
                concat_points = (boxes, box_labels)

        sparse_embeddings, dense_embeddings = self.sam_model.sam_prompt_encoder(
            points=concat_points,
            boxes=None,
            masks=mask_input,
        )

        # Predict masks
        batched_mode = (
            concat_points is not None and concat_points[0].shape[0] > 1
        )  # multi object prediction
        high_res_features = [feat_level for feat_level in self._features["high_res_feats"]]
        mask_embeddings, _, _, _ = self.sam_model.sam_mask_decoder(
            image_embeddings=self._features["image_embed"],
            image_pe=self.sam_model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=False,
            get_mask_embedding=True,
            high_res_features=high_res_features,
        )

        return mask_embeddings

    def forward(self, images, point=None, bboxes=None):
        device = next(self.sam_model.parameters()).device
        input_image = self._transforms(images).to(device)           # .bfloat16()
        batch_size, channel, image_size = input_image.shape[0], input_image.shape[1], input_image.shape[-2:]

        backbone_out = self.sam_model.forward_image(input_image)
        _, vision_feats, _, _ = self._prepare_backbone_features(backbone_out)
        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        if self.sam_model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.sam_model.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        self._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}

        # Get point prompts
        in_points = None
        points_scale = np.array(image_size)[::-1]
        if point is not None:
            points = [point]
            points = torch.as_tensor(
                points, dtype=torch.float32, device=device
            )
            in_points = self._transforms.transform_coords(
                points, normalize=True, orig_hw=image_size
            )
            num_points = in_points.shape[0]
        elif bboxes is not None:
            bboxes = torch.as_tensor(
                bboxes, dtype=torch.float, device=device
            )
            bboxes = self._transforms.transform_boxes(
                bboxes, normalize=True, orig_hw=image_size
            )  # Bx2x2
            num_points = bboxes.shape[0]
        else:
            # for training non-bboxes samples
            points = (self.point_grids[0] * points_scale)           # always 0, assuming no multi-levels
            points = torch.as_tensor(
                points, dtype=torch.float32, device=device
            )
            in_points = self._transforms.transform_coords(
                points, normalize=True, orig_hw=image_size
            )
            num_points = in_points.shape[0]

        # seems 1 for "include", 0 for "exclude", 2 (top-left corner) and 3 (bottom-right corner)
        in_labels = torch.ones(
            num_points, dtype=torch.int, device=device
        )

        # [B, D, Hf, Wf] (f: feature map)
        # [NOTE]: Can support both point and bboxes together
        if in_points is not None:
            mask_embeddings = self._predict(
                point_coords=in_points[None, :, :],
                point_labels=in_labels[None, :],
                boxes=None,
                multimask_output=self.multimask_output,
                return_logits=True,
            )
        elif bboxes is not None:
            mask_embeddings = self._predict(
                point_coords=None,
                point_labels=None,
                boxes=bboxes,
                multimask_output=self.multimask_output,
                return_logits=True,
            )
        else:
            raise ValueError("Either one of point or bboxes should be provided")

        mask_embeddings = self.sam2_reducer(mask_embeddings)
        patch_mask_embeddings = mask_embeddings.flatten(2).transpose(1,2)      # [B, Hf*Wf, D]
        patch_mask_embeddings = self.sam2vlm_layer(patch_mask_embeddings)
        return patch_mask_embeddings

    def _prepare_backbone_features(self, backbone_out):
        """Prepare and flatten visual features."""
        backbone_out = backbone_out.copy()
        assert len(backbone_out["backbone_fpn"]) == len(backbone_out["vision_pos_enc"])
        assert len(backbone_out["backbone_fpn"]) >= self.num_feature_levels

        feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels:]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels:]

        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
        # flatten NxCxHxW to HWxNxC
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]

        return backbone_out, vision_feats, vision_pos_embeds, feat_sizes


class MMEBModel(nn.Module):
    TRANSFORMER_CLS = AutoModelForCausalLM

    def __init__(self,
                 encoder: PreTrainedModel,
                 segmentation_model: torch.nn.Module = None,
                 pooling: str = 'last',
                 normalize: bool = False,
                 temperature: float = 0.02,
                 ):
        super().__init__()
        self.config = encoder.config
        self.encoder = encoder
        self.segmentation_model = segmentation_model

        self.pooling = pooling
        self.normalize = normalize
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.is_ddp = dist.is_initialized()
        if self.is_ddp:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    @property
    def device(self) -> torch.device:
        """Return the device of the first parameter."""
        return next(self.parameters()).device

    def print_parameter_count(self):
        print_master("========== Model parameter count: ========== ")

        """Print total and trainable parameters for the model."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print_master(f"Total parameters: {total_params:,}")
        print_master(f"Trainable parameters: {trainable_params:,}")
        print_master(f"Non-trainable parameters: {total_params - trainable_params:,}")
        
        # Also print_master breakdown by components
        print_master("\nParameter breakdown:")
        
        # Main encoder
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        encoder_trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        print_master(f"  Encoder: {encoder_params:,} total, {encoder_trainable:,} trainable")
        
        # Segmentation model (if exists)
        if self.segmentation_model is not None:
            sam_params = sum(p.numel() for p in self.segmentation_model.parameters())
            sam_trainable = sum(p.numel() for p in self.segmentation_model.parameters() if p.requires_grad)
            print_master(f"  SAM2: {sam_params:,} total, {sam_trainable:,} trainable")
        
        return total_params, trainable_params

    def encode_input(self, input):
        if getattr(self, "model_backbone", None) == INTERNVIDEO2:
            if "input_ids" in input.keys():
                # text side
                text_output = self.encoder.get_text_encoder()(
                    input["input_ids"],
                    attention_mask=input["attention_mask"],
                    return_dict=True,
                    mode="text",
                )
                text_embeds = text_output.last_hidden_state
                pooled_text_embeds = text_embeds[:, 0]
                pooled_output = self.encoder.text_proj(pooled_text_embeds)
                pooled_output /= pooled_output.norm(dim=-1, keepdim=True)
                return pooled_output
            else:
                _, vfeat = self.encoder.encode_vision(input["pixel_values"], test=True)
                vfeat = self.encoder.vision_proj(vfeat)
                vfeat /= vfeat.norm(dim=-1, keepdim=True)
                return vfeat
        elif getattr(self, "model_backbone", None) in [GME, LamRA, LamRA_QWEN2_5]:
            # pooled_output = self.encoder(**input, return_dict=True, output_hidden_states=True)
            texts = [text.replace(VLM_IMAGE_TOKENS[QWEN2_VL] + '\n', '') for text in input["texts"]] # we are actually passing video queries so this should not happen
            images = []
            for imgs in input['images']:
                # if multi images are given, select the middle frame only
                if isinstance(imgs, list):
                    imgs = imgs[len(imgs) // 2]
                    assert not isinstance(imgs, list) # make sure we have extracted the middle frame and it is no longer a list
                    images.append(imgs)
                else:
                    images.append(imgs)
            pooled_output = self.encoder.get_fused_embeddings(texts=texts, images=images)
            return pooled_output
        elif getattr(self, "model_backbone", None) == COLPALI:
            pooled_output = self.encoder(**input, return_dict=True, output_hidden_states=True)
            return pooled_output
        elif getattr(self, "model_backbone", None) == LLAVA_NEXT:
            input['pixel_values'] = input['pixel_values'].squeeze(dim=1)
            input['image_sizes'] = input['image_sizes'].squeeze(dim=1)
            hidden_states = self.encoder(**input, return_dict=True, output_hidden_states=True)
            hidden_states = hidden_states.hidden_states[-1]
            pooled_output = self._pooling(hidden_states, input['attention_mask'])
            return pooled_output
        else:
            hidden_states = self.encoder(**input, segmentation_model=self.segmentation_model, return_dict=True, output_hidden_states=True)
            hidden_states = hidden_states.hidden_states[-1]
            pooled_output = self._pooling(hidden_states, input['attention_mask'])
            return pooled_output

    def _pooling(self, last_hidden_state, attention_mask):
        if self.pooling == 'last' or self.pooling == 'eos':
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            batch_size = last_hidden_state.shape[0]
            if left_padding:
                # Get the vectors at the last position
                reps = last_hidden_state[torch.arange(batch_size), -1, :]
            else:
                # Calculate last 1 position in the original tensor
                eos_indices = attention_mask.sum(dim=1) - 1
                # Get the vectors at the last 1 position of each attention mask
                reps = last_hidden_state[
                    torch.arange(batch_size, device=last_hidden_state.device), eos_indices]
        else:
            raise NotImplementedError
        if self.normalize:
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps

    @classmethod
    def build(cls, model_args: ModelArguments, **kwargs):
        config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
        model_backbone = get_backbone_name(hf_config=config)
        print_master(f'Loading backbone [{model_backbone}] from {model_args.model_name}')
        # Loading the base model
        if model_backbone == PHI3V:
            config._attn_implementation = "eager"
            config.padding_side = "right"
            config.use_cache = False
            base_model = Phi3VForCausalLM.from_pretrained(
                model_args.model_name,
                config=config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
        elif model_backbone == LLAVA_NEXT:
            config.use_cache = False
            config.padding_side = "left"
            base_model = LlavaNextForConditionalGeneration.from_pretrained(
                model_args.model_name,
                config=config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
        elif model_backbone in [QWEN2_VL, QWEN2_5_VL]:
            config._attn_implementation = "flash_attention_2"
            config.padding_side = "left"
            config.use_cache = False
            base_model = backbone2model[model_backbone].from_pretrained(
                model_args.model_name,
                config=config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
        elif model_backbone in [QWEN2_VL_TOKENSELECTION, QWEN2_5_VL_TOKENSELECTION]:
            config._attn_implementation = "flash_attention_2"
            config.padding_side = "left"
            config.use_cache = False

            from .utils import parse_layer_type
            lm_qwen_layer = 28
            vis_qwen_layer = 32
            lm_skip_layer = parse_layer_type(model_args.lm_skip_layer, lm_qwen_layer)
            vis_skip_layer = parse_layer_type(model_args.vis_skip_layer, vis_qwen_layer)

            base_model = backbone2model[model_backbone].from_pretrained(
                model_args.model_name,
                config=config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                lm_skip_layer=lm_skip_layer,
                vis_skip_layer=vis_skip_layer,
            )
        else:
            config.use_cache = False
            base_model = cls.TRANSFORMER_CLS.from_pretrained(
                model_args.model_name, **kwargs, config=config,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True)
            
        # load SAM
        if model_args.sam:
            print_master(f"Loading SAM from {model_args.sam_config['checkpoint']}")
            model_args.sam_config["vlm_hidden_size"] = config.hidden_size
            sam2_model = SAM2VLM(model_args.sam_config)

            # Determine if it's a local path or HF Hub path
            checkpoint_filename = "sam2_reducer_and_vlm_layer.pth"
            if checkpoint_filename in list_repo_files(repo_id=config.name_or_path):
                # HF Hub path contains the checkpoint
                checkpoint_path = hf_hub_download(
                    repo_id=config.name_or_path,
                    filename=checkpoint_filename,
                )
                checkpoint = torch.load(checkpoint_path)
                sam2_model.sam2_reducer.load_state_dict(checkpoint['sam2_reducer'])
                sam2_model.sam2vlm_layer.load_state_dict(checkpoint['sam2vlm_layer'])
        else:
            sam2_model = None

        if model_args.lora:
            print_master(f'Loading lora adapter from {base_model}')
            lora_config = LoraConfig(
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,
                target_modules=model_args.lora_target_modules.split(','),
                lora_dropout=model_args.lora_dropout,
                init_lora_weights="gaussian",
                use_dora=True,
                inference_mode=False
            )
            lora_model = get_peft_model(base_model, lora_config)
            model = cls(
                encoder=lora_model,
                segmentation_model=sam2_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature
            )
        else:
            model = cls(
                encoder=base_model,
                segmentation_model=sam2_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature
            )
        
        # Print parameter count
        model.print_parameter_count()
        
        return model

    @classmethod
    def load(cls, model_args: ModelArguments, is_trainable=True, **kwargs):
        # Loading the base model
        model_name_or_path = model_args.checkpoint_path if model_args.checkpoint_path else model_args.model_name
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        if not hasattr(model_args, "model_backbone") or not model_args.model_backbone:
            model_backbone = get_backbone_name(hf_config=config, model_type=model_args.model_type)
            setattr(model_args, 'model_backbone', model_backbone)
        print_master(f'Loading backbone [{model_args.model_backbone}] from {model_name_or_path}')
        if model_args.model_backbone in {LLAVA_NEXT, QWEN2_VL, QWEN2_5_VL, QWEN2_VL_TOKENSELECTION, QWEN2_5_VL_TOKENSELECTION}:
            config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
            config._attn_implementation = "flash_attention_2"
            config.vision_config._attn_implementation = "flash_attention_2"
            base_model = backbone2model[model_args.model_backbone].from_pretrained(
                model_args.model_name,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                config=config
            )
        elif model_args.model_backbone == PHI3V:
            config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
            config.use_cache = False
            config.padding_side = "right"
            base_model = Phi3VForCausalLM.from_pretrained(model_args.model_name, **kwargs, config=config,
                                                          torch_dtype=torch.bfloat16, trust_remote_code=True)
            base_model.padding_side = "right"
        elif model_args.model_backbone == INTERNVIDEO2:
            print_master(f'Loading backbone [{model_args.model_backbone}] from {"src/model/vlm_backbone/internvideo2/"}')
            config = AutoConfig.from_pretrained("src/model/vlm_backbone/internvideo2/",
                                                trust_remote_code=True)
            base_model = backbone2model[model_args.model_backbone].from_pretrained("src/model/vlm_backbone/internvideo2/", config=config,
                                                                                   trust_remote_code=True)
        elif model_args.model_backbone == GME:
            base_model = GmeQwen2VL(model_args.model_name, processor=kwargs['processor'])
            setattr(base_model, 'config', config)
        elif model_args.model_backbone == LamRA:
            base_model = LamRAQwen2VL(model_args.model_name)
            setattr(base_model, 'config', config)
        elif model_args.model_backbone == LamRA_QWEN2_5:
            base_model = LamRAQwen25VL(model_args.model_name)
            setattr(base_model, 'config', config)
        elif model_args.model_backbone == COLPALI:
            base_model = ColPali.from_pretrained(model_args.model_name)
            setattr(base_model, 'config', config)
        else:
            # Loading external base model from HF
            config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
            config.use_cache = False
            base_model = cls.TRANSFORMER_CLS.from_pretrained(
                model_name_or_path, **kwargs, config=config,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True)
            
        # load SAM
        if model_args.sam:
            print_master(f"Loading SAM from {model_args.sam_config['checkpoint']}")
            model_args.sam_config["vlm_hidden_size"] = config.hidden_size
            sam2_model = SAM2VLM(model_args.sam_config)

            orig_sam_params = {
                name: param.clone().detach()
                for name, param in sam2_model.named_parameters()
                if param.requires_grad
            }

            # Determine if it's a local path or HF Hub path
            checkpoint_filename = "sam2_reducer_and_vlm_layer.pth"
            if os.path.exists(model_name_or_path):
                # Local checkpoint
                checkpoint_path = os.path.join(model_name_or_path, checkpoint_filename)
            elif checkpoint_filename in list_repo_files(repo_id=model_name_or_path):
                # HF Hub path contains the checkpoint
                checkpoint_path = hf_hub_download(
                    repo_id=model_name_or_path,
                    filename=checkpoint_filename,
                )
            else:
                raise FileNotFoundError(
                    f"{checkpoint_filename} not found in {model_name_or_path} "
                    "(not a local path with the file, nor in HF repo)"
                )

            checkpoint = torch.load(checkpoint_path)
            sam2_model.sam2_reducer.load_state_dict(checkpoint['sam2_reducer'])
            sam2_model.sam2vlm_layer.load_state_dict(checkpoint['sam2vlm_layer'])

            for name, param in sam2_model.named_parameters():
                if not param.requires_grad:
                    continue
                old = orig_sam_params[name]
                new = param.detach()
                delta = (new - old).abs().mean().item()
                print(f"{name:40s}  Δ mean abs = {delta:.6f}")
            print()
        else:
            sam2_model = None

        # Building the model on top of the base
        if model_args.lora:
            print_master(f'Loading LoRA from {model_name_or_path}')
            lora_config = LoraConfig.from_pretrained(model_name_or_path)
            lora_model = PeftModel.from_pretrained(base_model, model_name_or_path, config=lora_config, is_trainable=is_trainable)
            lora_model.load_adapter(model_name_or_path, lora_model.active_adapter, is_trainable=is_trainable)
            if not is_trainable:
                lora_model = lora_model.merge_and_unload()
            model = cls(
                encoder=lora_model,
                segmentation_model=sam2_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature
            )
        else:
            model = cls(
                encoder=base_model,
                segmentation_model=sam2_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature
            )

        model.model_backbone = model_args.model_backbone
        
        # Print parameter count
        model.print_parameter_count()
        
        return model

    def save(self, output_dir: str):
        self.encoder.save_pretrained(output_dir)

    def forward(self, qry: Dict[str, Tensor] = None, tgt: Dict[str, Tensor] = None, *args, **kwargs):
        qry_reps = self.encode_input(qry) if qry else None  # (bsz_per_device, dim)
        tgt_reps = self.encode_input(tgt) if tgt else None # (bsz_per_device, dim)

        if qry_reps is None or tgt_reps is None:
            return {"qry_reps": qry_reps, "tgt_reps": tgt_reps}

        if self.is_ddp:
            all_qry_reps = self._dist_gather_tensor(qry_reps)
            all_tgt_reps = self._dist_gather_tensor(tgt_reps)
        else:
            all_qry_reps = qry_reps
            all_tgt_reps = tgt_reps

        scores = self.compute_similarity(all_qry_reps, all_tgt_reps)
        scores = scores.view(all_qry_reps.size(0), -1)
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * (all_qry_reps.size(0) // all_tgt_reps.size(0))
        loss = self.cross_entropy(scores / self.temperature, target)
        if self.is_ddp:
            loss = loss * self.world_size

        return loss

    def _dist_gather_tensor(self, t: Tensor):
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)
        return all_tensors

    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))
