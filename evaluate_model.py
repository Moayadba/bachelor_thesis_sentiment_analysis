import pandas as pd
from sklearn.metrics import confusion_matrix
from utils.eval_utils import print_confusion_matrix
from sklearn.metrics import classification_report

eval_df = pd.read_excel("/Users/baset/Downloads/df_APPL_validation_aapl_only_model.xlsx")
LABELS = ['positive', 'negative', 'neutral']
eval_df['model_sentiment'] = eval_df['model_sentiment'].apply(lambda x: x.strip())
eval_df['sentiment'] = eval_df['sentiment'].apply(lambda x: x.lower())
# Plot confusion matrix
eval_df = eval_df[eval_df['model_sentiment'].isin(['positive', 'negative', 'neutral'])]
cnf_matrix = confusion_matrix(eval_df['sentiment'], eval_df['model_sentiment'], labels=['positive', 'negative', 'neutral'])
_ = print_confusion_matrix(cnf_matrix, LABELS)

# Print Precision Recall F1-Score Report

report = classification_report(eval_df['model_sentiment'], eval_df['sentiment'], target_names=LABELS)

print(report)