import pandas as pd
from sklearn.metrics import confusion_matrix
from utils.eval_utils import print_confusion_matrix
from sklearn.metrics import classification_report

FILE_WITH_PREDICTIONS_PATH = "/Users/baset/Downloads/GME_final_validation_sample_equal_dist_prediction_GPT3.csv"

def evaluate_model(df):

    LABELS = ['positive', 'negative', 'neutral']
    df['model_sentiment'] = df['model_sentiment'].apply(lambda x: x.strip())
    df['sentiment'] = df['sentiment'].apply(lambda x: x.lower())
    # Plot confusion matrix
    eval_df = df[df['model_sentiment'].isin(['positive', 'negative', 'neutral'])]
    cnf_matrix = confusion_matrix(eval_df['sentiment'], eval_df['model_sentiment'], labels=LABELS)
    _ = print_confusion_matrix(cnf_matrix, LABELS, figsize=(5, 4), fontsize=7)

    # Print Precision Recall F1-Score Report

    report = classification_report(eval_df['sentiment'], eval_df['model_sentiment'], labels=LABELS)
    print(cnf_matrix)
    print(report)
if __name__ == '__main__':
    df = pd.read_csv(FILE_WITH_PREDICTIONS_PATH)
    evaluate_model(df)
