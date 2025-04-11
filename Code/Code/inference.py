import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# loading the different models
model_paths = ['./Models/NER', './Models/Wiki+NER', './Models/Wiki_10', 'GroNLP/bert-base-dutch-cased', './Models/Wiki_Oneline' ,'./Models/ner_wiki_oneline']#, './Models/Wiki_NER_index']

for model_path in model_paths:
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, do_lower_case=True)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    
    model_path = os.path.basename(model_path).replace("/", "_")

    model.eval()

    def read_dataset(file_path):
        texts, labels = [], []
        current_text, current_labels = [], []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    token, label = line.split('\t')
                    current_text.append(token)
                    current_labels.append(label)
                elif current_text:
                    texts.append(current_text)
                    labels.append(current_labels)
                    current_text, current_labels = [], []
        
        if current_text:
            texts.append(current_text)
            labels.append(current_labels)
        
        return texts, labels

    # Using the NER dataset and retrieving a specific x sentence
    #Note: dataset_splits.json file is attached in the submission. This file contains the indices of the Validation and Test
    # data so that we can perform inference on sentences that were not seen by the model. To avoid data leakages and for robust results
    x = 5
    texts, labels = read_dataset('./Datasets//NER_data.txt')
    first_sentence = texts[x]

    # boolean for including/excluding the CLS and SEP token
    sep_token = [True, False]
    
    for sep in sep_token:
    
        inputs = tokenizer(first_sentence, return_tensors="pt", is_split_into_words=True, padding=True, truncation=True, add_special_tokens=sep)
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        # retrieving the attention weights and averaging the attention across heads
        attention = outputs.attentions[-1]
        avg_attention = attention.mean(dim=1).squeeze().numpy()

        # attention matrix
        plt.figure(figsize=(20, 16))
        sns.heatmap(avg_attention, xticklabels=tokens, yticklabels=tokens, cmap="YlOrRd", vmax = 0.6, vmin=0)
        plt.title(f"Model: {model_path}, SEP token: {sep}")
        plt.xlabel("Target Tokens")
        plt.ylabel("Source Tokens")
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        output_filename = f"{model_path}_SEP_token_{sep}.png"
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()

        # printing the predicted labels
        predictions = torch.argmax(outputs.logits, dim=-1)
        predicted_labels = [model.config.id2label[pred.item()] for pred in predictions[0]]

        print(f"Model: {model_path}, SEP token: {sep}")
        print("Tokens and Predicted Labels:}")
        for token, label in zip(tokens, predicted_labels):
            print(f"{token}: {label}")
        
        print()