import evaluate
import numpy as np


def compute_metrics(p):
    #Using seqeval for metrics 
    seqeval = evaluate.load("seqeval")
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    label_list = [
        "O",
        "B-ACTION",
        "I-ACTION",
        "B-TARGET",
        "I-TARGET",
        "B-CORRECTION",
        "I-CORRECTION",
    ]
    #getting y_pred and y_true
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    #giving results to huggingface trainer
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

