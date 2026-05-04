import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

def evaluate_performance(y_true, y_pred, y_prob):
    """
    Evaluates standard classification metrics.
    """
    metrics = {}
    
    # Cast to ensure proper typing
    try:
        y_true = np.array(y_true, dtype=int)
        y_pred = np.array(y_pred, dtype=int)
        y_prob = np.array(y_prob, dtype=float)
    except Exception:
        return {'accuracy': np.nan, 'auc': np.nan, 'f1': np.nan, 'auprc': np.nan}
        
    if len(y_true) == 0:
        return {'accuracy': np.nan, 'auc': np.nan, 'f1': np.nan, 'auprc': np.nan}

    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    try:
        if len(np.unique(y_true)) > 1:
            metrics['auc'] = roc_auc_score(y_true, y_prob)
            metrics['auprc'] = average_precision_score(y_true, y_prob)
        else:
            metrics['auc'] = np.nan
            metrics['auprc'] = np.nan
            
        metrics['f1'] = f1_score(y_true, y_pred, average='macro' if len(np.unique(y_true)) > 2 else 'binary')
    except ValueError:
        metrics['auc'] = np.nan
        metrics['f1'] = np.nan
        metrics['auprc'] = np.nan
        
    return metrics
