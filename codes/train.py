import transformers.trainer as _tr; _tr.check_torch_load_is_safe = lambda: None
####### BREAK TORCH LOAD ISSUE < 2.6 disable the CVE‐enforced check #######

import os
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

# Adapted from Tevatron code
import logging
import os.path
import hydra
import sys
import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from transformers import HfArgumentParser
from src.arguments import ModelArguments, DataArguments, TrainingArguments
from src.data.collator.train_collator import MultimodalDataCollator
from src.data.loader.mixed_dataset import init_mixed_dataset
from src.model.model import MMEBModel
from src.trainer import GradCacheLateProcessTrainer
from src.utils import print_rank, print_master, find_latest_checkpoint
from src.model.processor import load_processor, get_backbone_name


logging.basicConfig(
    level=logging.INFO, format='[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]  # Ensures logs appear in stdout
)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="./configs", config_name="vlm2vec_train")
def main(config: DictConfig):
    config = OmegaConf.to_container(config, resolve=True)
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    config['output_dir'] = os.path.join(hydra.core.hydra_config.HydraConfig.get()['runtime']['output_dir'], config['output_dir'])
    config['logging_dir'] = os.path.join(hydra.core.hydra_config.HydraConfig.get()['runtime']['output_dir'], config['output_dir'])
    print_master(f"========= Save path: {config['logging_dir']} =========")

    model_args, data_args, training_args = parser.parse_dict(config)
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    # Check for existing checkpoints
    if training_args.resume_from == 'auto':
        resume_checkpoint_dir = find_latest_checkpoint(training_args.output_dir)
        if resume_checkpoint_dir:
            logger.info(f"========= Resuming from checkpoint: {resume_checkpoint_dir}")
    elif training_args.resume_from.isdigit():
        resume_checkpoint_dir = os.path.join(training_args.output_dir, f'checkpoint-{training_args.resume_from}')
        if os.path.exists(resume_checkpoint_dir):
            logger.info(f"========= Resuming from checkpoint: {resume_checkpoint_dir}")
    elif "checkpoint" in training_args.resume_from:
        resume_checkpoint_dir = training_args.resume_from
        if os.path.exists(resume_checkpoint_dir):
            logger.info(f"========= Resuming from checkpoint: {resume_checkpoint_dir}")
    else:
        resume_checkpoint_dir = None
        logger.info("========= No checkpoint found. Starting fresh training.")

    model = MMEBModel.build(model_args)
    model_backbone = get_backbone_name(hf_config=model.config)
    setattr(model_args, 'model_backbone', model_backbone)
    setattr(training_args, 'model_backbone', model_backbone)
    print_rank(f'model_backbone: {model_backbone}')
    processor = load_processor(model_args, data_args)
    setattr(model, 'processor', processor)

    with open(data_args.dataset_config, 'r') as yaml_file:
        dataset_config = yaml.safe_load(yaml_file)
        train_dataset = init_mixed_dataset(dataset_config, model_args, data_args, training_args)
    train_collator = MultimodalDataCollator(processor, model_args, data_args, training_args)

    trainer_cls = GradCacheLateProcessTrainer
    trainer = trainer_cls(
        model=model,
        processing_class=processor,
        args=training_args,
        model_args=model_args,
        train_dataset=train_dataset,
        data_collator=train_collator,
        max_length=data_args.max_len,
    )
    train_dataset.trainer = trainer

    trainer.train(resume_from_checkpoint=resume_checkpoint_dir)
    trainer.save_model(training_args.output_dir)

    if trainer.is_world_process_zero():
        processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    import traceback
    from hydra.core.global_hydra import GlobalHydra

    if GlobalHydra.instance().is_initialized():
        print("--- Hydra was already initialized by: ---")
        traceback.print_stack()
        GlobalHydra.instance().clear()

    main()
