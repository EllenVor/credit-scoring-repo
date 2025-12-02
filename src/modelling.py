import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns   
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, roc_auc_score

def dataset_scaler(data):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    scaled_df = pd.DataFrame(scaled_data, columns=data.columns)
    return scaled_df

def importance_graph(feature_names, importances):

  model_coefficients = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
  model_coefficients = model_coefficients.sort_values(by='Importance', ascending=False)   
  plt.figure(figsize=(10, 6))
  sns.barplot(x='Importance', y='Feature', data=model_coefficients) 
  
def plot_roc_curve(y_scores, X_test, y_test):

  # Compute ROC curve and ROC AUC score
  fpr, tpr, thresholds = roc_curve(y_test, y_scores)
  auc = roc_auc_score(y_test, y_scores)
  
  # Plot
  plt.figure(figsize=(8, 6))
  plt.plot(fpr, tpr, label=f'Logistic ROC (AUC = {auc:.3f})', linewidth=2)
  plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1)
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('ROC Curve - Logistic Regression')
  plt.legend(loc='lower right')
  plt.grid(True)
  plt.show()




