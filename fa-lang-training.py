from pathlib import Path
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from transformers import RobertaConfig, RobertaTokenizerFast, RobertaForMaskedLM, LineByLineTextDataset, DataCollatorForLanguageModeling, TrainingArguments, Trainer, pipeline
import os

model_path = "./FarBERTo"
# Getting all the TXT dataset in the project
paths = [str(x) for x in Path(".").glob("**/*.txt")]

tokenizer_train = ByteLevelBPETokenizer()

# customizing the training
tokenizer_train.train(files=paths, vocab_size=100_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

# save trained tokenizer model
os.mkdir(model_path)
tokenizer_train.save_model(model_path, "FarBERTo")


# For testing the model

tokenizer = ByteLevelBPETokenizer(
    "./FarBERTo/vocab.json",
    "./FarBERTo/merges.txt",
)

tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)
tokenizer.enable_truncation(max_length=512)
tokenizer.encode("سلام، من یاشا هستم. علی رو میشناسی؟")

# Training a model for the tokenizer
config = RobertaConfig(
    vocab_size=100_000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

model_tokenizer = RobertaTokenizerFast.from_pretrained("./FarBERTo", max_len=512)
model = RobertaForMaskedLM(config=config)

# dataset
dataset = LineByLineTextDataset(
    tokenizer=model_tokenizer,
    file_path="./dataset/farsi.txt",  # TODO - should be changed to the real path of the dataset
    block_size=128,
)


data_collator = DataCollatorForLanguageModeling(
    tokenizer=model_tokenizer, mlm=True, mlm_probability=0.15
)

# Last step
training_args = TrainingArguments(
    output_dir="./EsperBERTo",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_gpu_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    prediction_loss_only=True,
)

trainer.train()
trainer.save_model(model_path)

# test the model

fill_mask = pipeline(
    "fill-mask",
    model="./FarBERTo",
    tokenizer="./FarBERTo"
)

fill_mask("علی با دوستش بیرون <mask>")
