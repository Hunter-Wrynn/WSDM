import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import torch
from time import time
from datasets import load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments,
    AutoModelForSequenceClassification,
    Trainer,
    EarlyStoppingCallback
)
from trl import SFTTrainer,setup_chat_format
import numpy as np
from transformers import BitsAndBytesConfig, AutoModelForSequenceClassification
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoTokenizer
from datasets import DatasetDict, Dataset

class CFG:
    VER = 1
    NUM_LABELS = 3
    BATCH_SIZE = 4
    EPOCHS = 1
    
    MODEL_NAME = 'meta-llama/Llama-3.1-8B'

    
    SEED = 2024 
    MAX_LENGTH = 1024 
    
    OUTPUT_DIR = 'Llama 3.1 8b fine-tuned model'
    

model = AutoModelForSequenceClassification.from_pretrained(
    CFG.MODEL_NAME,
    quantization_config=quantization_config,
    num_labels=CFG.NUM_LABELS,
    device_map='auto'
)