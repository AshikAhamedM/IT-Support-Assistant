from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from it_support_prompts import format_it_prompt

MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

datasets = [
    load_dataset("json", data_files="data/samples/password_reset.jsonl")["train"],
    load_dataset("json", data_files="data/samples/vpn_issues.jsonl")["train"],
    load_dataset("json", data_files="data/samples/email_issues.jsonl")["train"],
]

dataset = concatenate_datasets(datasets)

tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(example):
    text = format_it_prompt(example["instruction"], example["output"])
    return tokenizer(text, truncation=True, padding="max_length", max_length=512)

dataset = dataset.map(tokenize)

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    load_in_8bit=True,
    device_map="auto"
)

lora = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora)

args = TrainingArguments(
    output_dir="it-support-lora",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    fp16=True,
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset
)

trainer.train()
model.save_pretrained("it-support-lora-model")
tokenizer.save_pretrained("it-support-lora-model")
