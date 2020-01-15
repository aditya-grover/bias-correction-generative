import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (brier_score_loss, precision_score, recall_score, f1_score, mean_squared_error)
from sklearn.calibration import calibration_curve
import os
import numpy as np
from matplotlib import animation

from sklearn.utils import column_or_1d
from sklearn.preprocessing import label_binarize


def extended_calibration_curve(y_true, y_prob, normalize=False, n_bins=5, strategy='uniform'):
    """Compute true and predicted probabilities for a calibration curve.
     The method assumes the inputs come from a binary classifier.
     Calibration curves may also be referred to as reliability diagrams.
    Read more in the :ref:`User Guide <calibration>`.
    Parameters
    ----------
    y_true : array, shape (n_samples,)
        True targets.
    y_prob : array, shape (n_samples,)
        Probabilities of the positive class.
    normalize : bool, optional, default=False
        Whether y_prob needs to be normalized into the bin [0, 1], i.e. is not
        a proper probability. If True, the smallest value in y_prob is mapped
        onto 0 and the largest one onto 1.
    n_bins : int
        Number of bins. A bigger number requires more data.
    strategy : {'uniform', 'quantile'}, (default='uniform')
        Strategy used to define the widths of the bins.
        uniform
            All bins have identical widths.
        quantile
            All bins have the same number of points.
    Returns
    -------
    prob_true : array, shape (n_bins,)
        The true probability in each bin (fraction of positives).
    prob_pred : array, shape (n_bins,)
        The mean predicted probability in each bin.
    normalized_bin_cnts : array, shape (n_bins,)
        The density of each bin.
    """
    y_true = column_or_1d(y_true)
    y_prob = column_or_1d(y_prob)

    if normalize:  # Normalize predicted values into interval [0, 1]
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())
    elif y_prob.min() < 0 or y_prob.max() > 1:
        raise ValueError("y_prob has values outside [0, 1] and normalize is "
                         "set to False.")

    labels = np.unique(y_true)
    if len(labels) > 2:
        raise ValueError("Only binary classification is supported. "
                         "Provided labels %s." % labels)
    y_true = label_binarize(y_true, labels)[:, 0]

    if strategy == 'quantile':  # Determine bin edges by distribution of data
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(y_prob, quantiles * 100)
        bins[-1] = bins[-1] + 1e-8
    elif strategy == 'uniform':
        bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
    else:
        raise ValueError("Invalid entry to 'strategy' input. Strategy "
                         "must be either 'quantile' or 'uniform'.")

    bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1

    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0
    prob_true = (bin_true[nonzero] / bin_total[nonzero])
    prob_pred = (bin_sums[nonzero] / bin_total[nonzero])

    return prob_true, prob_pred, bin_total[nonzero]/np.sum(bin_total[nonzero])

def plot_calibration(fraction_of_positives, mean_predicted_value, savefile=None):

    fig = plt.figure(figsize=(5, 4))
    ax = plt.axes(xlim=(-0.05, 1.05), ylim=(-0.05, 1.05))

    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    ax.plot(mean_predicted_value, fraction_of_positives, "s-")
    ax.set_ylabel("Observed fraction of positives")
    ax.set_xlabel("Expected fraction of positives")
    ax.legend(loc="lower right")
    ax.set_title('Calibration plots  (reliability curve)')

    if savefile:
        plt.savefig(savefile)
    else:
        plt.show()
    plt.close()

    return


def plot_calibration_curve(y_test, y_pred, prob_pos, name, savedir=None, verbose=False):
    # print(y_test.shape, y_pred.shape, prob_pos.shape)
    # print(y_test[:10], y_pred[:10], prob_pos[:10])
    
    num_bins = 10
    fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_pos, n_bins=num_bins)

    calibration_error = mean_squared_error(fraction_of_positives, mean_predicted_value)
    clf_score = brier_score_loss(y_test, prob_pos)

    if verbose:
        print("%s:" % name)
        print("\tBrier: %1.3f" % (clf_score))
        print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
        print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
        print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))

    fig = plt.figure(figsize=(5, 4))
    ax1 = plt.axes(xlim=(-0.05, 1.05), ylim=(-0.05, 1.05))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label="%s (%1.3f)" % (name, clf_score))
    ax1.set_ylabel("Observed fraction of positives")
    ax1.set_xlabel("Expected fraction of positives")
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    if savedir:
        plt.savefig(os.path.join(savedir, name))
        print(os.path.join(savedir, name))
    else:
        plt.show()
    plt.close()
    return clf_score, calibration_error

