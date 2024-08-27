# from sklearn.utils import resample
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset, Subset
# from sklearn.model_selection import KFold
# from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report
# import pandas as pd
# import numpy as np
# from transformers import BertTokenizer, BertForSequenceClassification

# # Hyperparameters
# lr = 2e-05
# epochs = 10
# batch_size = 16
# k_folds = 2

# # Load dataset
# foods = pd.read_csv('menu-items-dataset-subset.csv')
# columns_to_fill = ['Name', 'Description', 'Category', 'Category Description', 'Modifier Group', 'Modifier Group Description']
# foods[columns_to_fill] = foods[columns_to_fill].fillna('')

# possible_labels = ["Gluten Free", "Vegan", "Vegetarian", "Contains Nuts"]
# thresholds = {'Gluten Free': 0.50, 'Vegan': 0.50, 'Vegetarian': 0.50, 'Contains Nuts': 0.50}

# class SentenceDataset(Dataset):
#     def __init__(self, database):
#         self.database = database
#         # Balance the dataset here
#         self.database = self._balance_dataset(self.database)

#     def _balance_dataset(self, df):
#         balanced_df = pd.DataFrame()
#         for label in possible_labels:
#             majority_class = df[df[label] == 0]
#             minority_class = df[df[label] == 1]
            
#             # Upsample minority class
#             minority_upsampled = resample(minority_class,
#                                         replace=True,
#                                         n_samples=len(majority_class),
#                                         random_state=42)
            
#             # Combine majority class with upsampled minority class
#             balanced_df = pd.concat([balanced_df, majority_class, minority_upsampled])
        
#         return balanced_df


#     def __len__(self):
#         return len(self.database)

#     def __getitem__(self, idx):
#         row = self.database.iloc[idx]
#         text = str(row["Name"]) + "," + str(row["Description"]) + "," + str(row["Category"]) + "," + str(row["Category Description"]) + "," + str(row["Modifier Group"]) + "," + str(row["Modifier Group Description"])
#         label = row[possible_labels].values.astype(float)
#         return text, label

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# # Initialize tokenizer and model
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# def calculate_metrics(labels, predictions, confidences):
#     precision = precision_score(labels, predictions, average='weighted', zero_division=0)
#     recall = recall_score(labels, predictions, average='weighted', zero_division=0)
#     f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
#     roc_auc = roc_auc_score(labels, confidences, average='weighted', multi_class='ovr')
#     return precision, recall, f1, roc_auc

# def train(model, iterator, optimizer, criterion, device):
#     model.train()
#     train_loss = 0
#     for sentences, labels in iterator:
#         encoding = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
#         input_ids = encoding['input_ids'].to(device)
#         attention_mask = encoding['attention_mask'].to(device)
#         labels = labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(input_ids, attention_mask=attention_mask)
#         loss = criterion(outputs.logits, labels)
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()
#     return train_loss / len(iterator)

# def test(model, iterator, criterion, device, thresholds):
#     model.eval()
#     all_labels = []
#     all_predictions = []
#     all_confidences = []
#     total_loss = 0
#     with torch.no_grad():
#         for sentences, labels in iterator:
#             encoding = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
#             input_ids = encoding['input_ids'].to(device)
#             attention_mask = encoding['attention_mask'].to(device)
#             labels = labels.to(device)
#             outputs = model(input_ids, attention_mask=attention_mask)
#             prob = outputs.logits.sigmoid()
#             loss = criterion(outputs.logits, labels)
#             total_loss += loss.item()
            
#             # Apply thresholds for each label
#             predictions = torch.zeros_like(prob)
#             for i, label in enumerate(possible_labels):
#                 predictions[:, i] = (prob[:, i] > thresholds[label]).float()
            
#             all_labels.append(labels.cpu().numpy())
#             all_predictions.append(predictions.cpu().numpy())
#             all_confidences.append(prob.cpu().numpy())
    
#     all_labels = np.vstack(all_labels)
#     all_predictions = np.vstack(all_predictions)
#     all_confidences = np.vstack(all_confidences)
    
#     accuracy = np.mean((all_predictions == all_labels).sum() / all_labels.size)
#     precision, recall, f1, roc_auc = calculate_metrics(all_labels, all_predictions, all_confidences)
    
#     # Print classification report for each label separately
#     for i, label in enumerate(possible_labels):
#         labels_for_report = all_labels[:, i]
#         predictions_for_report = all_predictions[:, i]
#         print(f'Classification Report for {label}:')
#         print(classification_report(labels_for_report, predictions_for_report, labels=[0, 1], target_names=[f'Not {label}', label]))
    
#     return total_loss / len(iterator), accuracy, precision, recall, f1, roc_auc

# class EarlyStopping:
#     def __init__(self, patience=2, min_delta=0):
#         self.patience = patience
#         self.min_delta = min_delta
#         self.counter = 0
#         self.best_loss = None
#         self.early_stop = False

#     def __call__(self, val_loss):
#         if self.best_loss is None:
#             self.best_loss = val_loss
#         elif val_loss < self.best_loss - self.min_delta:
#             self.best_loss = val_loss
#             self.counter = 0
#         else:
#             self.counter += 1
#             if self.counter >= self.patience:
#                 self.early_stop = True

# dataset = SentenceDataset(foods)
# kfold = KFold(n_splits=k_folds, shuffle=True)

# # Calculate class weights based on your data
# class_weights = torch.tensor([1.0 / (sum(foods[label] == 1) / len(foods)) for label in possible_labels]).to(device)

# # K-Fold Cross-Validation
# fold_results = []

# for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
#     print(f'===== Fold {fold + 1} / {k_folds} =====')
    
#     train_subsampler = Subset(dataset, train_ids)
#     test_subsampler = Subset(dataset, test_ids)

#     train_loader = DataLoader(train_subsampler, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_subsampler, batch_size=batch_size, shuffle=False)
    
#     model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(possible_labels))
#     model.to(device)
    
#     optimizer = optim.AdamW(model.parameters(), lr=lr)
#     criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)

#     train_losses = []
#     valid_losses = []

#     early_stopping = EarlyStopping(patience=2, min_delta=0)

#     for e in range(epochs):
#         print(f"Epoch {e + 1}/{epochs}")
#         train_loss = train(model, train_loader, optimizer, criterion, device)
#         valid_loss, accuracy, precision, recall, f1, roc_auc = test(model, test_loader, criterion, device, thresholds)
#         train_losses.append(train_loss)
#         valid_losses.append(valid_loss)
#         print(f"Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Acc: {accuracy:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")
        
#         # Early stopping
#         early_stopping(valid_loss)
#         if early_stopping.early_stop:
#             print("Early stopping")
#             break

#     fold_results.append({
#         'train_losses': train_losses,
#         'valid_losses': valid_losses,
#         'accuracy': accuracy,
#         'precision': precision,
#         'recall': recall,
#         'f1': f1,
#         'roc_auc': roc_auc
#     })
    
#     # Save the model for each fold
#     model_save_path = f'model_fold_{fold + 1}.pt'
#     torch.save(model.state_dict(), model_save_path)
#     print(f"Model saved to {model_save_path}")

# # Calculate average metrics over all folds
# avg_accuracy = np.mean([res['accuracy'] for res in fold_results])
# avg_precision = np.mean([res['precision'] for res in fold_results])
# avg_recall = np.mean([res['recall'] for res in fold_results])
# avg_f1 = np.mean([res['f1'] for res in fold_results])
# avg_roc_auc = np.mean([res['roc_auc'] for res in fold_results])

# print(f"\nAverage Cross-Validation Metrics:\nAccuracy: {avg_accuracy:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1: {avg_f1:.4f}, ROC-AUC: {avg_roc_auc:.4f}")

# # Get final predictions and confidences
# # def get_predictions(model, dataset, device, thresholds):
# #     model.eval()
# #     all_predictions = []
# #     all_confidences = []

# #     with torch.no_grad():
# #         for idx in range(len(dataset)):
# #             sentence, _ = dataset[idx]
# #             encoding = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
# #             input_ids = encoding['input_ids'].to(device)
# #             attention_mask = encoding['attention_mask'].to(device)

# #             outputs = model(input_ids, attention_mask=attention_mask)
# #             prob = outputs.logits.sigmoid()

# #             prediction = []
# #             for i, label in enumerate(possible_labels):
# #                 prediction.append((prob[0, i] > thresholds[label]).float().item())

# #             all_predictions.append(prediction)
# #             all_confidences.append(prob.cpu().numpy().flatten())

# #     return np.array(all_predictions), np.array(all_confidences)

# # predictions, confidences = get_predictions(final_model, dataset, device, thresholds)

# # # Add predictions and confidences to the dataframe
# # for i, label in enumerate(possible_labels):
# #     foods[f'{label}_Prediction'] = predictions[:, i]
# #     foods[f'{label}_Confidence'] = confidences[:, i]

# # # Save the dataframe with predictions and confidences
# # foods.to_csv('menu-items-predictions.csv', index=False)
# # print("Predictions and confidences saved to menu-items-predictions.csv")


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.utils import resample
import seaborn as sns
import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler('output.log'),  # Log to file
                        logging.StreamHandler()             # Log to console
                    ])
logger = logging.getLogger()

# Hyperparameters
lr = 1.6e-05
weight_decay = 5e-04
epochs = 50
batch_size = 16
k_folds = 5

# Load dataset
foods = pd.read_csv('training-data2.csv')
columns_to_fill = ['name', 'description', 'menu_category_name', 'menu_category_description', 'modifier_group_name', 'modifier_group_description']
foods[columns_to_fill] = foods[columns_to_fill].fillna('')

possible_labels = ["vegan", "vegetarian", "gluten_free", "contains_nuts"]
thresholds = {'vegan': 0.5, 'vegetarian': 0.5, 'gluten_free': 0.5, 'contains_nuts': 0.5}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class MultiLabelDataset(Dataset):
    def __init__(self, database):
        self.labels = ["vegan", "vegetarian", "gluten_free", "contains_nuts"]
        self.database = self._balance_dataset(database)

    def __len__(self):
        return len(self.database)

    def __getitem__(self, idx):
        row = self.database.iloc[idx]
        # text = str(row["Name"]) + "," + str(row["Description"]) + "," + str(row["Category"]) + "," + str(row["Category Description"]) + "," + str(row["Modifier Group"]) + "," + str(row["Modifier Group Description"])
        text = str(row["name"]) + "," + str(row["description"]) + "," + str(row["menu_category_name"]) + "," + str(row["menu_category_description"]) + "," + str(row["modifier_group_name"]) + "," + str(row["modifier_group_description"])
        labels = torch.tensor([row[label] for label in self.labels], dtype=torch.float)
        return text, labels

    def _balance_dataset(self, df):
        balanced_df = df.copy()
        for label in self.labels:
            majority_class = df[df[label] == 0]
            minority_class = df[df[label] == 1]
            # Upsample minority class
            minority_upsampled = resample(minority_class,
                                          replace=True,
                                          n_samples=len(majority_class),
                                          random_state=42)
            # Combine majority class with upsampled minority class
            balanced_df = pd.concat([balanced_df, minority_upsampled])
        return balanced_df

def calculate_metrics(labels, predictions, confidences):
    precision = precision_score(labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(labels, predictions, average='weighted', zero_division=0)
    f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
    roc_auc = roc_auc_score(labels, confidences, average='weighted')
    return precision, recall, f1, roc_auc

def train(model, iterator, optimizer, criterion, device):
    model.train()
    train_loss = 0
    for sentences, labels in iterator:
        encoding = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(iterator)

def test(model, iterator, criterion, device):
    model.eval()
    all_labels = []
    all_predictions = []
    all_confidences = []
    total_loss = 0
    with torch.no_grad():
        for sentences, labels in iterator:
            encoding = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            labels = labels.to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            prob = outputs.logits.sigmoid()
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()

            predictions = (prob > 0.5).float()
            
            all_labels.append(labels.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())
            all_confidences.append(prob.cpu().numpy())

    all_labels = np.vstack(all_labels)
    all_predictions = np.vstack(all_predictions)
    all_confidences = np.vstack(all_confidences)

    accuracy = np.mean((all_predictions == all_labels).sum(axis=1) / all_labels.shape[1])
    precision, recall, f1, roc_auc = calculate_metrics(all_labels, all_predictions, all_confidences)

    # Print confusion matrices for all labels
    for idx, label in enumerate(possible_labels):
        true_label = all_labels[:, idx]
        pred_label = all_predictions[:, idx]
        print_confusion_matrix(true_label, pred_label, label)

    return total_loss / len(iterator), accuracy, precision, recall, f1, roc_auc

def print_confusion_matrix(y_true, y_pred, label):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    logger.info(f"Confusion Matrix for {label}:")
    logger.info(f"True Negative (TN): {cm[0, 0]}")
    logger.info(f"False Positive (FP): {cm[0, 1]}")
    logger.info(f"False Negative (FN): {cm[1, 0]}")
    logger.info(f"True Positive (TP): {cm[1, 1]}")
    logger.info(f"Normalized Confusion Matrix:\n{cm_normalized}")
    
    print(f"Confusion Matrix for {label}:")
    print(f"True Negative (TN): {cm[0, 0]}")
    print(f"False Positive (FP): {cm[0, 1]}")
    print(f"False Negative (FN): {cm[1, 0]}")
    print(f"True Positive (TP): {cm[1, 1]}")
    print(f"Normalized Confusion Matrix:\n{cm_normalized}")

class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.counter = 0
        self.best_f1 = -float('inf')  # Initialize with negative infinity
        self.early_stop = False
        self.best_model_state = None
        self.save_path = ''

    def __call__(self, f1, model_state_dict, save_path):
        if f1 >= self.best_f1:  # Only save if the F1 score is better
            self.best_f1 = f1
            self.counter = 0
            self.save_path = save_path
            torch.save(model_state_dict, save_path)
            self.best_model_state = model_state_dict
            logging.info(f"Updated Model with F1: {f1:.4f}")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                logging.info("Early stopping triggered")
    
    def load_best_model(self):
        return torch.load(self.save_path)

def main():
    dataset = MultiLabelDataset(foods)
    kfold = KFold(n_splits=k_folds, shuffle=True)

    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        logger.info(f'===== Fold {fold + 1} / {k_folds} =====')

        train_subsampler = Subset(dataset, train_ids)
        test_subsampler = Subset(dataset, test_ids)

        train_loader = DataLoader(train_subsampler, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_subsampler, batch_size=batch_size, shuffle=False)

        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(possible_labels))
        model.to(device)

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.BCEWithLogitsLoss()

        train_losses = []
        valid_losses = []

        early_stopping = EarlyStopping(patience=5)

        for e in range(epochs):
            logger.info(f"Epoch {e + 1}/{epochs}")
            train_loss = train(model, train_loader, optimizer, criterion, device)
            valid_loss, accuracy, precision, recall, f1, roc_auc = test(model, test_loader, criterion, device)
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            logger.info(f"Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Acc: {accuracy:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")

            # Early stopping
            early_stopping(f1, model.state_dict(), save_path=f'model_fold_{fold + 1}_best_40k_f1:{f1}_epoch:{e}.pt')
            if early_stopping.early_stop:
                logger.info("Early stopping")
                break

        # Load the best model for evaluation
        model.load_state_dict(early_stopping.load_best_model())
        valid_loss, accuracy, precision, recall, f1, roc_auc = test(model, test_loader, criterion, device)
        
        # Log fold results
        logger.info(f"\nFold Results:")
        logger.info(f"Train Loss: {train_losses[-1]:.4f}, Valid Loss: {valid_losses[-1]:.4f}, Acc: {accuracy:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")

if __name__ == "__main__":
    main()

