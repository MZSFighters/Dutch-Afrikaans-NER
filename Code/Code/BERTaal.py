import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import wandb
import json

wandb.init(project="NLP_Project", name="Wiki-NER-30e-b16-2e-5")

# hyperparams
num_epochs = 30
learning_rate = 2e-5
batch_size = 16
max_len = 128
patience = 3  # early stopping
min_delta = 0.001

# labels for NER dataset
label2id = {'out': 0, 'b-org': 1, 'i-org': 2, 'b-loc': 3, 'i-loc': 4, 'b-pers': 5, 'i-pers': 6, 'b-misc': 7, 'i-misc': 8}
id2label = {v: k for k, v in label2id.items()}

class NERDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        labels = self.labels[item]

        encoding = self.tokenizer(text, 
                                  is_split_into_words=True, 
                                  return_offsets_mapping=True, 
                                  padding='max_length', 
                                  truncation=True, 
                                  max_length=self.max_len)

        word_ids = encoding.word_ids()
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            else:
                label_ids.append(label2id[labels[word_idx]])

        encoding['labels'] = label_ids
        encoding['input_ids'] = torch.tensor(encoding['input_ids'])
        encoding['attention_mask'] = torch.tensor(encoding['attention_mask'])
        encoding['labels'] = torch.tensor(encoding['labels'])

        return encoding

# function that reads the dataset
def read_dataset(file_path):
    texts, labels = [], []
    current_text, current_labels = [], []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                token, label = line.split('\t')
                current_text.append(token.lower())
                current_labels.append(label.lower())
            elif current_text:
                texts.append(current_text)
                labels.append(current_labels)
                current_text, current_labels = [], []
    
    if current_text:
        texts.append(current_text)
        labels.append(current_labels)
    
    return texts, labels


texts, labels = read_dataset('./NER_data.txt')
indices = list(range(len(texts)))

# train-val-test split
train_texts, temp_texts, train_labels, temp_labels, train_indices, temp_indices = train_test_split(
    texts, labels, indices, test_size=0.3, random_state=42)

val_texts, test_texts, val_labels, test_labels, val_indices, test_indices = train_test_split(
    temp_texts, temp_labels, temp_indices, test_size=0.5, random_state=42)


# Loading the model and the tokenizer
tokenizer = AutoTokenizer.from_pretrained('Wiki_10', do_lower_case=True)
model = AutoModelForTokenClassification.from_pretrained('Wiki_10', 
                                                        num_labels=len(label2id),
                                                        id2label=id2label,
                                                        label2id=label2id)

# keeping track of indices of the validation and test sentences
split_indices = {
    'validation_indices': val_indices,
    'test_indices': test_indices
}
with open('dataset_splits.json', 'w') as f:
    json.dump(split_indices, f)

print("\nValidation set indices:", val_indices)
print("\nTest set indices:", test_indices)

# using dataloader for loading the train-val-test data
train_dataset = NERDataset(train_texts, train_labels, tokenizer, max_len)
val_dataset = NERDataset(val_texts, val_labels, tokenizer, max_len)
test_dataset = NERDataset(test_texts, test_labels, tokenizer, max_len)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = AdamW(model.parameters(), lr=learning_rate)


# logging hyperparameters to wandb
wandb.config.update({
    "learning_rate": learning_rate,
    "epochs": num_epochs,
    "batch_size": batch_size,
    "model": "bert-base-dutch-cased",
    "max_length": max_len,
    "early_stopping_patience": patience,
    "early_stopping_min_delta": min_delta
})

best_val_loss = float('inf')
epochs_without_improvement = 0
best_model = None

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    
    avg_train_loss = total_loss / len(train_loader)
    wandb.log({"train_loss": avg_train_loss}, step=epoch)

    # Validation
    model.eval()
    val_loss = 0
    val_predictions, val_true_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            val_loss += outputs.loss.item()
            
            logits = outputs.logits
            pred = torch.argmax(logits, dim=2)
    
            for i, label in enumerate(labels):
                pred_i = pred[i, labels[i] != -100]
                label_i = labels[i, labels[i] != -100]
                val_predictions.extend([id2label[p.item()] for p in pred_i])
                val_true_labels.extend([id2label[l.item()] for l in label_i])

    avg_val_loss = val_loss / len(val_loader)
    val_report = classification_report(val_true_labels, val_predictions, output_dict=True)
    
    wandb.log({
        "val_loss": avg_val_loss,
        "val_f1": val_report['weighted avg']['f1-score'],
        "val_precision": val_report['weighted avg']['precision'],
        "val_recall": val_report['weighted avg']['recall']
    }, step=epoch)

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Training Loss: {avg_train_loss:.4f}")
    print(f"Validation Loss: {avg_val_loss:.4f}")
    print("Validation Performance:")
    print(classification_report(val_true_labels, val_predictions))

    # only saving the best model
    if avg_val_loss < best_val_loss - min_delta:
        best_val_loss = avg_val_loss
        epochs_without_improvement = 0
        best_model = model.state_dict()
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:
        print(f"Early stopping triggered after {epoch+1} epochs")
        break

# best model if early stopping was triggered
if best_model is not None:
    model.load_state_dict(best_model)


# Final evaluation on test set
model.eval()
test_predictions, test_true_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        
        logits = outputs.logits
        pred = torch.argmax(logits, dim=2)
        
        for i, label in enumerate(labels):
            pred_i = pred[i, labels[i] != -100]
            label_i = labels[i, labels[i] != -100]
            test_predictions.extend([id2label[p.item()] for p in pred_i])
            test_true_labels.extend([id2label[l.item()] for l in label_i])

print("\nFinal Test Set Performance:")
test_report = classification_report(test_true_labels, test_predictions, output_dict=True)
print(classification_report(test_true_labels, test_predictions))

wandb.log({
    "test_f1": test_report['weighted avg']['f1-score'],
    "test_precision": test_report['weighted avg']['precision'],
    "test_recall": test_report['weighted avg']['recall']
})

# Saving the fine-tuned model and its tokenizer
model.save_pretrained("./Wiki_NER_index")
tokenizer.save_pretrained("./Wiki_NER_index")
print("Fine-tuning complete. Model saved.")
wandb.finish()