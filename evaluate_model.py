import pandas as pd
from sklearn.metrics import confusion_matrix
from utils.eval_utils import print_confusion_matrix
from sklearn.metrics import classification_report

eval_df = pd.read_csv("/Users/baset/Downloads/sample_5_new_prediction_101_200_gpt_ada.csv")
correct = 0
wrong = 0
neutral = 0
LABELS = ['positive', 'negative', 'neutral']
eval_df['model_sentiment'] = eval_df['model_sentiment'].apply(lambda x: x.strip())
eval_df['sentiment'] = eval_df['sentiment'].apply(lambda x: x.lower())
# Plot confusion matrix
eval_df = eval_df[eval_df['model_sentiment'].isin(['positive', 'negative', 'neutral'])]
cnf_matrix = confusion_matrix(eval_df['model_sentiment'], eval_df['sentiment'])
_ = print_confusion_matrix(cnf_matrix, LABELS)

# Print Precision Recall F1-Score Report


report = classification_report(eval_df['model_sentiment'], eval_df['sentiment'], target_names=LABELS)

print(report)