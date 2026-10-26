import json
from transformers import AutoTokenizer
from datasets import Dataset

with open("token_classification_data.json") as f:
    payload = json.load(f)

label_list = payload["label_list"]
examples = payload["data"]
# retrieving dataset
ds = Dataset.from_list([
    {"tokens": ex["tokens"], "ner_tags": ex["ner_tags"]}
    for ex in examples
]) 

# get tokenizer from huggingface
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
 
 
def tokenize_and_align_labels(batch):
    # pass tokens to tokenizer
    tokenized_inputs = tokenizer(
        batch["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = []
    # check if word is part of original text, if not append -100 to tell huggingface to ignore the label
    for i, label in enumerate(batch["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)                 
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)                        
            previous_word_idx = word_idx
        all_labels.append(label_ids)
    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs
 
 # write out to a datasets .arrow 
tokenized_ds = ds.map(tokenize_and_align_labels, batched=True)
 
example = tokenized_ds[0]
tokens = tokenizer.convert_ids_to_tokens(example["input_ids"])
print("Subword tokens:", tokens)
print("Aligned labels:", example["labels"])
print("(-100 = ignored by loss; everything else is an index into label_list)")
print("label_list:", label_list)
 
tokenized_ds.save_to_disk("mario_token_classification_tokenized")
print("\nSaved tokenized dataset to ./mario_token_classification_tokenized")
