import argparse
import logging
import os
import sys
import shutil
import time
import boto3
import tarfile
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, SFTConfig

# --- Setup Logging ---
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--hf_token_env", type=str, default="HF_TOKEN")
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--train_dir", type=str, default="/opt/ml/input/data/train")
    parser.add_argument("--eval_dir", type=str, default="/opt/ml/input/data/eval")
    
    # --- FIX: Use Home Directory (Guaranteed Writable) for temporary model save ---
    home_dir = os.path.expanduser("~")
    unique_id = int(time.time())
    safe_output_path = os.path.join(home_dir, f"hrai_model_{unique_id}")
    
    parser.add_argument("--output_dir", type=str, default=safe_output_path)
    
    # ANTI-OVERFITTING: Stick to 1.0 epoch for robust training
    parser.add_argument("--epochs", type=float, default=1.0) 

    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    
    # ANTI-OVERFITTING: Lower Learning Rate for better generalization
    parser.add_argument("--learning_rate", type=float, default=1e-5) 

    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--lora_r", type=int, default=16)

    # ANTI-OVERFITTING: Reduce Alpha from 32 -> 16
    parser.add_argument("--lora_alpha", type=int, default=16)

    # ANTI-OVERFITTING: Increase Dropout from 0.05 -> 0.1
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    
    # --- CRITICAL FIX: High-frequency Evaluation/Saving for best model selection ---
    parser.add_argument("--eval_steps", type=int, default=10) 
    parser.add_argument("--save_steps", type=int, default=10)

    args, unknown = parser.parse_known_args()
    return args

def load_hf_token(token_env_var):
    token = os.environ.get(token_env_var)
    if not token:
        logger.warning(f"Missing HF token env var: {token_env_var}")
    return token

def load_training_datasets(train_dir, eval_dir, tokenizer):
    logger.info(f"Loading dataset from {train_dir} and {eval_dir}")
    train_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith(".jsonl")]
    eval_files = [os.path.join(eval_dir, f) for f in os.listdir(eval_dir) if f.endswith(".jsonl")]
    data_files = {"train": train_files[0], "eval": eval_files[0]}
    dataset = load_dataset("json", data_files=data_files)

    def format_chat(example):
        text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        return {"text": text}

    logger.info("Formatting dataset...")
    # Using num_proc=os.cpu_count() is safe here because we are on the p4d.24xlarge instance with massive RAM
    dataset = dataset.map(format_chat, num_proc=os.cpu_count(), remove_columns=dataset["train"].column_names)
    return dataset["train"], dataset["eval"]

def upload_to_s3(local_path, s3_bucket, s3_prefix):
    """
    Directly zips and uploads the model to S3, bypassing SageMaker permissions.
    """
    logger.info(f"--- STARTING DIRECT S3 UPLOAD ---")
    logger.info(f"Source: {local_path}")
    logger.info(f"Destination: s3://{s3_bucket}/{s3_prefix}/model.tar.gz")
    
    # We compress the entire contents of the output directory
    tar_path = os.path.join(os.path.dirname(local_path), "model.tar.gz")
    
    # 1. Compress
    logger.info("Compressing model...")
    with tarfile.open(tar_path, "w:gz") as tar:
        # NOTE: tar.add ensures the checkpoint folder structure is preserved
        tar.add(local_path, arcname=".")
        
    # 2. Upload
    logger.info("Uploading to S3...")
    s3 = boto3.client('s3')
    s3.upload_file(tar_path, s3_bucket, f"{s3_prefix}/model.tar.gz")
    logger.info("✅ UPLOAD SUCCESS! Your model is safe in S3.")

def main():
    args = parse_args()
    token = load_hf_token(args.hf_token_env)
    
    logger.info(f"Loading tokenizer: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=token)
    if tokenizer.pad_token is None: tokenizer.pad_token = "<|reserved_special_token_0|>"
    tokenizer.padding_side = "right"

    logger.info("Configuring QLoRA...")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, quantization_config=bnb_config, torch_dtype=torch.bfloat16, device_map="auto", token=token, trust_remote_code=True)
    
    # LoRA config uses hyperparameters designed to fight overfitting
    peft_config = LoraConfig(
        r=args.lora_r, 
        lora_alpha=args.lora_alpha, 
        lora_dropout=args.lora_dropout, 
        bias="none", 
        task_type="CAUSAL_LM", 
        target_modules="all-linear"
    )
    
    train_dataset, eval_dataset = load_training_datasets(args.train_dir, args.eval_dir, tokenizer)

    logger.info(f"Setting SFTConfig (Output: {args.output_dir})")
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        
        # CRITICAL SETTINGS FOR BEST MODEL SELECTION
        eval_strategy="steps",        
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        load_best_model_at_end=True, # Ensure the best model is saved
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        disable_tqdm=False,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = SFTTrainer(model=model, peft_config=peft_config, tokenizer=tokenizer, args=sft_config, train_dataset=train_dataset, eval_dataset=eval_dataset)

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training Complete")

    # --- SAVE THE BEST MODEL FOUND BY THE TRAINER ---
    # The trainer finds the best checkpoint based on eval_loss and saves it to output_dir
    logger.info(f"Saving final/best model to: {args.output_dir}")
    trainer.save_model(args.output_dir)
    
    # --- BYPASS: Direct S3 Upload ---
    bucket_name = os.environ.get("SM_TRAINING_BUCKET", "sagemaker-us-east-1-937127308917")
    # This uses the specific job name that SageMaker assigned to the fit() call
    job_name = os.environ.get("TRAINING_JOB_NAME", f"manual-upload-{int(time.time())}")
    
    # The final model is saved to S3
    upload_to_s3(args.output_dir, bucket_name, f"{job_name}/output")

if __name__ == "__main__":
    main()