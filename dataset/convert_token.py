import json
 
SRC = "output.json"
OUT = "token_classification_data.json"
 
with open(SRC) as f:
    data = json.load(f)
 
examples = data["rasa_nlu_data"]["common_examples"]

# taking action, target and correction labels from output.json and converting them to token classification tags
tag2label = {'action':"B-ACTION",'target':"B-TARGET",'correction_connector': "B-CORRECTION"}
tag2labelcontinue = {'action':"I-ACTION",'target':"I-TARGET",'correction_connector': "I-CORRECTION"}
label_list = ["O", "B-ACTION", "I-ACTION", "B-TARGET", "I-TARGET","B-CORRECTION","I-CORRECTION"]
label2id = {l: i for i, l in enumerate(label_list)}

converted = []
for ex in examples:
    # extracting tagged text
    entities = ex['entities']
    sentence = ex['text']
    words = sentence.split()
    tags = []
    last_tag = 'None'
    for w in words:
        
        for entity in entities:
            # checking if word matches any words in tagged phrase
            if w in entity['value'].split():
                if entity['entity'] not in last_tag:
                    # label with B if not the same tag as the last tagged label
                    last_tag = entity['entity']
                    tags.append(tag2label[entity['entity']])
                    break
                tags.append(tag2labelcontinue[entity['entity']])
                break
        else:
            # if no labels, append 0 and set last_tag to None
            tags.append("O")
            last_tag = 'None'
    # append sorted tokens to converted 
    converted.append({
        "tokens": words,
        "ner_tags": [label2id[t] for t in tags],
        "ner_tags_str": tags,  
        # drop before training
        "intent": ex["intent"], 
        # kept for reference
    })

# write out to a json file
with open(OUT, "w") as f:
    json.dump({
        "label_list": label_list,
        "label2id": label2id,
        "id2label": {v: k for k, v in label2id.items()},
        "data": converted,
    }, f, indent=2)
 
print(f"Wrote {len(converted)} examples to {OUT}")
print("\nSample:")
print(json.dumps(converted[0], indent=2))
print(json.dumps(converted[300], indent=2))
