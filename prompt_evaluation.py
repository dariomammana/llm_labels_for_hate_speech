import pandas as pd
import numpy as np
import krippendorff
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

language = "SI" 
temperature = "biased_personas"  # or "redefined_class1" or "0.2" or "1"

path_to_sampled_labels = f"Data/Sample150_{language}.csv"

# --- Load Sample150_{language}.csv ---
sample = pd.read_csv(path_to_sampled_labels)
print(f"Sample150_{language}.csv structure:")
print(sample.head())
print(f"Columns: {sample.columns.tolist()}")
print()

# --- Load {language}_prompt_{temperature}.csv ---
prompt_results = pd.read_csv(f"{language}_prompt_{temperature}.csv")
print(f"{language}_prompt_{temperature}.csv structure:")
print(prompt_results.head())
print(f"Columns: {prompt_results.columns.tolist()}")
print()

# --- Compute Krippendorff's ordinal alpha for {language}_prompt_{temperature}.csv ---
# Format: each row has label_run1 and label_run2
# Krippendorff's alpha expects data as a list of lists (coders x items)
labels_run1 = prompt_results['label_run1'].values
labels_run2 = prompt_results['label_run2'].values

# Convert to numeric (handle ERROR values)
def convert_to_numeric(labels):
    result = []
    for label in labels:
        if label == 'ERROR':
            result.append(np.nan)  # Missing value
        else:
            result.append(float(label))
    return result

labels_run1_numeric = convert_to_numeric(labels_run1)
labels_run2_numeric = convert_to_numeric(labels_run2)

# Prepare data for krippendorff_alpha: shape (num_coders, num_items)
data_prompt = np.array([labels_run1_numeric, labels_run2_numeric])

# Compute ordinal alpha
alpha_prompt = krippendorff.alpha(data_prompt, level_of_measurement='ordinal')
print(f"Krippendorff's Ordinal Alpha ({language}_prompt_{temperature}.csv): {alpha_prompt:.4f}")
print()

# --- Compute Krippendorff's ordinal alpha for Sample150_{language}.csv ---
# Each row has type_1 and type_2 annotations
type1_numeric = sample['type_1'].values.astype(float)
type2_numeric = sample['type_2'].values.astype(float)

# Prepare data for krippendorff_alpha: shape (num_coders, num_items)
data_sample = np.array([type1_numeric, type2_numeric])

alpha_sample = krippendorff.alpha(data_sample, level_of_measurement='ordinal')
print(f"Krippendorff's Ordinal Alpha (Sample150_{language}.csv): {alpha_sample:.4f}")
print(f"Number of paired annotations: {len(sample)}")
print()

# --- Compute overall agreement between Sample150_{language}.csv and {language}_prompt_{temperature}.csv ---
# Merge the two files on text to get all 4 annotations per comment
merged = sample.merge(prompt_results, left_on='text', right_on='comment', how='inner')
print(f"Matched comments between Sample150_{language}.csv and {language}_prompt_{temperature}.csv: {len(merged)}")

if len(merged) > 0:
    # Extract all 4 annotations: type_1, type_2, label_run1, label_run2
    type1 = merged['type_1'].values.astype(float)
    type2 = merged['type_2'].values.astype(float)
    
    # Handle ERROR values in prompt results
    def safe_convert(labels):
        result = []
        for label in labels:
            if label == 'ERROR':
                result.append(np.nan)
            else:
                result.append(float(label))
        return result
    
    run1 = np.array(safe_convert(merged['label_run1'].values))
    run2 = np.array(safe_convert(merged['label_run2'].values))
    
    # Prepare data for krippendorff_alpha: shape (num_coders, num_items)
    # 4 coders: human1 (type_1), human2 (type_2), prompt1 (label_run1), prompt2 (label_run2)
    data_overall = np.array([type1, type2, run1, run2])
    
    alpha_overall = krippendorff.alpha(data_overall, level_of_measurement='ordinal')
    print(f"Krippendorff's Ordinal Alpha (Overall - Humans vs Prompt): {alpha_overall:.4f}")
else:
    print(f"Could not merge Sample150_{language}.csv and {language}_prompt_{temperature}.csv")

print()

# --- Create Confusion Matrix ---
if len(merged) > 0:
    # Flatten all labels: use both type_1 and type_2 from sample, both label_run1 and label_run2 from prompt
    sample_labels = np.concatenate([
        merged['type_1'].astype(float).values,
        merged['type_2'].astype(float).values
    ]).astype(int)
    
    # Handle ERROR values for prompt
    prompt_labels_list = []
    for idx, row in merged.iterrows():
        run1 = 0 if row['label_run1'] == 'ERROR' else float(row['label_run1'])
        run2 = 0 if row['label_run2'] == 'ERROR' else float(row['label_run2'])
        prompt_labels_list.append(int(run1))
        prompt_labels_list.append(int(run2))
    prompt_labels = np.array(prompt_labels_list)
    
    # Create confusion matrix
    cm = confusion_matrix(sample_labels, prompt_labels, labels=[0, 1, 2, 3])
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['0: Appropriate', '1: Inappropriate', '2: Offensive', '3: Violent'],
                yticklabels=['0: Appropriate', '1: Inappropriate', '2: Offensive', '3: Violent'],
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix: Sample150_{language} vs {language}_prompt_{temperature}\n(300 labels total: Human on Y-axis, Claude on X-axis)', fontsize=12)
    plt.ylabel('Human Annotations (300 labels)', fontsize=11)
    plt.xlabel('Claude Annotations (300 labels)', fontsize=11)
    plt.tight_layout()
    
    # Save and show
    output_filename = f"confusion_matrix_{language}_{temperature}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to: {output_filename}")
    plt.show()
