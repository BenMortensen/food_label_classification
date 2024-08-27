import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the unified model
model_filename = 'model_fold_4_best_40k.pt'
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)  # Adjust num_labels based on your setup
model.load_state_dict(torch.load(model_filename, map_location=device))
model.to(device)
model.eval()

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define the labels and thresholds
labels = ['vegan', 'vegetarian', 'gluten_free', 'contains_nuts']
thresholds = {'vegan': 0.99, 'vegetarian': 0.99, 'gluten_free': 0.99, 'contains_nuts': 0.99}

def preprocess(text):
    encoding = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    return input_ids, attention_mask

def predict(text):
    input_ids, attention_mask = preprocess(text)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        probabilities = outputs.logits.sigmoid().cpu().numpy()[0]
    return probabilities

def interpret_predictions(probabilities, thresholds):
    results = {}
    for i, label in enumerate(labels):
        probability = probabilities[i]
        if probability > thresholds[label]:
            results[label] = (True, probability * 100)  # Confidence in positive prediction
        else:
            results[label] = (False, (1 - probability) * 100)  # Confidence in negative prediction
    return results

def process_csv(input_csv, output_csv):
    # Load the new data
    data = pd.read_csv(input_csv)
    
    # Ensure the relevant columns are filled with empty strings if NaN
    # columns_to_fill = ['Name', 'Description', 'Category', 'Category Description', 'Modifier Group', 'Modifier Group Description']
    columns_to_fill = ['name', 'description', 'menu_category_name', 'menu_category_description', 'modifier_group_name', 'modifier_group_description']
    data[columns_to_fill] = data[columns_to_fill].fillna('')

    # Initialize statistics storage
    stats = {label: {'mismatches': 0, 'false_positives': 0, 'false_negatives': 0,
                     'fp_confidences': [], 'fn_confidences': []} for label in labels}

    # Add columns for predictions, confidence percentages, and mismatches
    for label in labels:
        data[f"{label} Predicted"] = ''
        data[f"{label} Confidence (%)"] = ''
        data[f"{label} Mismatch"] = ''

    # Iterate over each row and make predictions
    for idx, row in data.iterrows():
        text = str(row["name"]) + "," + str(row["description"]) + "," + str(row["menu_category_name"]) + "," + str(row["menu_category_description"]) + "," + str(row["modifier_group_name"]) + "," + str(row["modifier_group_description"])
        probabilities = predict(text)
        results = interpret_predictions(probabilities, thresholds)

        for label in labels:
            prediction, confidence = results[label]
            data.at[idx, f"{label} Predicted"] = prediction
            data.at[idx, f"{label} Confidence (%)"] = confidence

            # Determine if there is a mismatch
            actual_label = row[label]  # Assuming the actual label is a boolean (True/False)
            if actual_label != prediction:
                data.at[idx, f"{label} Mismatch"] = True
                stats[label]['mismatches'] += 1

                # False Positive: actual label is False, but prediction is True
                if not actual_label and prediction:
                    stats[label]['false_positives'] += 1
                    stats[label]['fp_confidences'].append(confidence)
                
                # False Negative: actual label is True, but prediction is False
                if actual_label and not prediction:
                    stats[label]['false_negatives'] += 1
                    stats[label]['fn_confidences'].append(confidence)
            else:
                data.at[idx, f"{label} Mismatch"] = False

    # Calculate and print statistics
    for label in labels:
        num_fp = stats[label]['false_positives']
        num_fn = stats[label]['false_negatives']
        avg_fp_conf = sum(stats[label]['fp_confidences']) / len(stats[label]['fp_confidences']) if stats[label]['fp_confidences'] else 0
        avg_fn_conf = sum(stats[label]['fn_confidences']) / len(stats[label]['fn_confidences']) if stats[label]['fn_confidences'] else 0
        
        print(f"Label: {label}")
        print(f"  Mismatches: {stats[label]['mismatches']}")
        print(f"  False Positives: {num_fp}, Average Confidence: {avg_fp_conf:.2f}%")
        print(f"  False Negatives: {num_fn}, Average Confidence: {avg_fn_conf:.2f}%")

    # Define the desired column order
    columns_order = [
        'id', 'name', 'description',
        'vegan', 'vegan Predicted', 'vegan Confidence (%)', 'vegan Mismatch',
        'vegetarian', 'vegetarian Predicted', 'vegetarian Confidence (%)', 'vegetarian Mismatch',
        'gluten_free', 'gluten_free Predicted', 'gluten_free Confidence (%)', 'gluten_free Mismatch',
        'contains_nuts', 'contains_nuts Predicted', 'contains_nuts Confidence (%)', 'contains_nuts Mismatch',
        'menu_category_name', 'menu_category_description', 'modifier_group_name', 'modifier_group_description'
    ]

    # Reorder the DataFrame columns
    data = data[columns_order]

    # Save the DataFrame with the predictions and confidence percentages to a new CSV file
    data.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")

# Example usage
process_csv('training-data2.csv', 'predicted_menu_items_with_confidence.csv')