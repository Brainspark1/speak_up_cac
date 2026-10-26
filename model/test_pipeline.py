# test pipeline for quickly testing model on sentences
from transformers import pipeline
text = input('Type something: ')
classifier = pipeline("ner", model="Saggarwal/GAMEBERT")
print(classifier(text))
