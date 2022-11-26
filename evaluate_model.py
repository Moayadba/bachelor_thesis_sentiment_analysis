import pandas as pd
from sklearn.metrics import confusion_matrix
from utils.eval_utils import print_confusion_matrix
from sklearn.metrics import classification_report

eval_df = pd.read_csv("/Users/baset/Desktop/Kursanis Thesis/Datasets/validatation/GME_verify_2.csv")

LABELS = ['positive', 'negative', 'neutral']
eval_df['model_sentiment'] = eval_df['model_sentiment'].apply(lambda x: x.strip())
eval_df['sentiment'] = eval_df['sentiment'].apply(lambda x: x.lower())
# Plot confusion matrix
eval_df = eval_df[eval_df['model_sentiment'].isin(['positive', 'negative', 'neutral'])]
cnf_matrix = confusion_matrix(eval_df['sentiment'], eval_df['model_sentiment'], labels=LABELS)
_ = print_confusion_matrix(cnf_matrix, LABELS, figsize=(5, 4), fontsize=7)

# Print Precision Recall F1-Score Report

report = classification_report(eval_df['sentiment'], eval_df['model_sentiment'], labels=LABELS)

print(report)