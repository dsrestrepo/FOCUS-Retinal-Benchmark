import numpy as np

def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    binids = np.minimum((y_prob * n_bins).astype(int), n_bins - 1)
    
    ece = 0.0
    for i in range(n_bins):
        mask = binids == i
        if np.sum(mask) > 0:
            prob_mean = np.mean(y_prob[mask])
            acc_mean = np.mean(y_true[mask])
            ece += np.abs(prob_mean - acc_mean) * np.sum(mask) / len(y_true)
    return ece

def evaluate_calibration(y_true, y_prob):
    metrics = {}
    try:
        y_true = np.array(y_true, dtype=int)
        y_prob = np.array(y_prob, dtype=float)
        
        if len(y_true) == 0 or len(np.unique(y_true)) < 2:
            return {'ece': np.nan}
            
        metrics['ece'] = expected_calibration_error(y_true, y_prob)
    except Exception:
        metrics['ece'] = np.nan
        
    return metrics
