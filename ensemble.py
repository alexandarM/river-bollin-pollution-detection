# ensemble.py
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

from config import DEVICE, BATCH_SIZE
from dataset_utils import SimpleDataset, AE_TF


def compute_ae_errors(model, data_loader, device=DEVICE):
    #Compute reconstruction errors for all samples in data_loader
    model.eval()
    errors = []
    with torch.no_grad():
        for imgs, _ in data_loader:
            imgs = imgs.to(device)
            err = model.reconstruction_error(imgs).cpu().numpy()
            errors.extend(err)
    return np.array(errors)


def normalize_scores(scores):
    """Min-max normalization to [0,1]."""
    min_s = scores.min()
    max_s = scores.max()
    if max_s - min_s < 1e-8:
        return np.zeros_like(scores)
    return (scores - min_s) / (max_s - min_s)


def ensemble_scores(ae_errors, clf_probs, ae_weight=0.5):
    """
    Combine AE error (anomaly score) and classifier probability.
    ae_errors: higher = more anomalous (pollution)
    clf_probs: higher = more likely pollution
    Returns ensemble score (0-1, higher = pollution).
    """
    norm_ae = normalize_scores(ae_errors)
    norm_clf = normalize_scores(clf_probs)
    return ae_weight * norm_ae + (1 - ae_weight) * norm_clf


def evaluate_ensemble(ae_model, clf_model, test_paths, positive_set, device=DEVICE):
    """
    Evaluate ensemble on test set.
    Returns ensemble scores, true labels, and metrics.
    """
    # Get classifier probabilities
    from classifier import evaluate_classifier
    from dataset_utils import RiverDataset, BASE_TF
    test_dataset = RiverDataset(test_paths, positive_set, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    clf_metrics = evaluate_classifier(clf_model, test_loader, device)
    clf_probs = clf_metrics['probs']
    true_labels = clf_metrics['labels']

    # Get AE errors
    ae_loader = DataLoader(SimpleDataset(test_paths, AE_TF), batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    ae_errors = compute_ae_errors(ae_model, ae_loader, device)

    # Ensemble
    ensemble_scores_ = ensemble_scores(ae_errors, clf_probs, ae_weight=0.5)

    # Metrics
    ensemble_metrics = {
        'auc': roc_auc_score(true_labels, ensemble_scores_),
        'avg_precision': average_precision_score(true_labels, ensemble_scores_),
        'precision': None, 'recall': None, 'thresholds': None
    }
    precision, recall, thresholds = precision_recall_curve(true_labels, ensemble_scores_)
    ensemble_metrics['precision'] = precision
    ensemble_metrics['recall'] = recall
    ensemble_metrics['thresholds'] = thresholds

    return ensemble_scores_, true_labels, ensemble_metrics, clf_metrics, ae_errors
