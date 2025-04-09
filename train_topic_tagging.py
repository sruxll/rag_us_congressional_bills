import torch
import accelerate
from datasets import load_dataset
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)

# remove problematic keyword arguments if present, which found during debugging
_original_accelerator_init = accelerate.Accelerator.__init__
def new_accelerator_init(self, *args, **kwargs):
    for key in ['dispatch_batches', 'even_batches', 'use_seedable_sampler']:
        kwargs.pop(key, None)
    _original_accelerator_init(self, *args, **kwargs)
accelerate.Accelerator.__init__ = new_accelerator_init

# set up the device to accommodate difference: use the MPS backend if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# choose smallest Flan-T5
model_name = "google/flan-t5-small"

# load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.to(device)

# load the dataset from Hugging Face
dataset = load_dataset("dreamproit/bill_labels_us")

# define maximum sequence lengths for inputs and targets
max_input_length = 512
max_target_length = 64

# # -------------- inspect the raw data length ----------------
# lengths = [len(tokenizer(x, return_tensors="pt")["input_ids"][0]) for x in dataset["train"]["text"]]
# print(min(lengths), np.median(lengths), max(lengths))

# preprocessing function to tokenize inputs (bill text) and targets (policy area)
def preprocess_function(examples):
    inputs = examples["text"]
    targets = examples["policy_area"]
    # Tokenize the input text (bill content)
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    # Tokenize the target labels (policy tags)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# tokenize the entire dataset using multiple processes for speedup
tokenized_dataset = dataset.map(preprocess_function, batched=True, num_proc=4)

# in case the dataset doesn't have an explicit 'train' and 'test' split
if "train" not in tokenized_dataset.keys() or "test" not in tokenized_dataset.keys():
    tokenized_dataset = tokenized_dataset["train"].train_test_split(test_size=0.1)

# for MPS, disable mixed precision flags (bf16/fp16)
if device.type == "mps":
    training_args = Seq2SeqTrainingArguments(
        output_dir="./flan-t5-finetuned-bill_labels",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        weight_decay=0.01,
        save_strategy="epoch",
        predict_with_generate=True,
        fp16=False,
    )
else:
    # bf16 for supported CPU devices 
    training_args = Seq2SeqTrainingArguments(
        output_dir="./flan-t5-finetuned-bill_labels",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        weight_decay=0.01,
        save_strategy="epoch",
        predict_with_generate=True,
        bf16=True,  
    )

# create a data collator that dynamically pads the inputs
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# initialize the Seq2SeqTrainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# start training
trainer.train()

# save the fine-tuned model and tokenizer
model.save_pretrained("./flan-t5-finetuned-bill_labels")
tokenizer.save_pretrained("./flan-t5-finetuned-bill_labels")