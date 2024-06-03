import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from scipy.stats import entropy
from typing import Type, Any, Callable, Union, List, Optional

from utilis.utils import normalized


def acc_cal(logits, label: List[int], method: str = 'top1'):
    if method == 'top1':
        label_pred = np.argsort(logits, -1).T[-1]
        correct = np.sum([i == j for i, j in zip(label_pred, label)])
        total = len(label)

    result = correct / total * 100
    return round(float(result), 4)


def ClsAccCal(logits, label: List[int], method: str = 'top1'):
    if method == 'top1':
        label_pred = np.argsort(logits, -1).T[-1]

        label_set = list(set(label))
        correct = np.zeros(len(label_set))
        total = np.zeros(len(label_set)) + 1e-6
        for i, j in zip(label_pred, label):
            correct[j] += (i == j)
            total[j] += 1

    result = np.array(correct) / np.array(total) * 100
    return result.round(4).tolist()


def mmf_acc_cal(logits, label: List[int], class_num_list: List[int], method: str = 'top1'):
    # many medium few - shot accuracy calculation
    correct = np.zeros(3)
    total = np.zeros(3) + 1e-6
    mmf_id = list(map(get_mmf_idx, list(class_num_list)))
    if method == 'top1':
        label_pred = np.argsort(logits, -1).T[-1]
        for i, j in zip(label_pred, label):
            correct[mmf_id[j]] += (i == j)
            total[mmf_id[j]] += 1

    result = np.array(correct) / np.array(total) * 100
    return result.round(4).tolist()


def get_mmf_idx(img_num: int):
    assert type(img_num) == int
    if img_num < 20:
        return 2
    elif img_num < 100:
        return 1
    else:
        return 0


# Original Code from https://github.com/gpleiss/temperature_scaling
class ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """

    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, softmaxes, labels):
        softmaxes = torch.Tensor(softmaxes)
        labels = torch.LongTensor(labels)

        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=softmaxes.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


# Expected Calibration Error (ECE) numpy implementation
def ECECal(probs, labels: List[int], bins: int = 15, sum=True):
    conf = np.max(probs, axis=-1)
    acc = np.argmax(probs, axis=-1) == labels

    bin_upper_bounds = np.linspace(0, 1, bins + 1)[1:]
    # split_idx = np.searchsorted(bin_upper_bounds, conf, 'left')
    split_idx = np.digitize(conf, bin_upper_bounds, right=True)
    data_len = len(split_idx)

    ece = np.zeros(bins)
    for i in range(bins):
        idx = split_idx == i
        if np.sum(idx) > 0:
            bin_avg_conf = np.mean(conf[idx])
            bin_avg_acc = np.mean(acc[idx])
            bin_prob = np.sum(idx) / data_len

            ece[i] = np.abs(bin_avg_acc - bin_avg_conf) * bin_prob

        # print(bin_avg_acc, bin_avg_conf, bin_prob, ece[i])

    return ece.sum()


# Reliability Values numpy implementation
def ECEAccCal(probs, labels, bins: int = 15):
    conf = np.max(probs, axis=-1)
    acc = np.argmax(probs, axis=-1) == labels

    bin_upper_bounds = np.linspace(0, 1, bins + 1)[1:]
    # split_idx = np.searchsorted(bin_upper_bounds, conf, 'left')
    split_idx = np.digitize(conf, bin_upper_bounds, right=True)
    data_len = len(labels)

    bin_acc = np.zeros(bins)
    bin_prob = np.zeros(bins)
    for i in range(bins):
        idx = split_idx == i
        if np.sum(idx) > 0:
            bin_avg_acc = np.mean(acc[idx])
            bin_acc[i] = bin_avg_acc
            bin_prob[i] = np.sum(idx) / data_len

        # print(bin_avg_acc, bin_avg_conf, bin_prob, ece[i])

    return bin_acc, bin_prob


def BierCal(probs, labels: List[int]):
    probs_correct = np.array([x[i] for x, i in zip(probs, labels)])
    return np.mean(np.power(probs_correct - 1, 2))


def EntropyCal(prob):
    result = np.mean(entropy(prob, axis=-1))
    return result


def SCECal(probs, labels: List[int], bins: int = 15):
    cls = list(set(labels))
    conf_all = np.max(probs, axis=-1)
    acc_all = np.argmax(probs, axis=-1) == labels

    conf_group, acc_group = group_data((conf_all, acc_all), labels, cls)

    eces = []
    for conf, acc in zip(conf_group, acc_group):
        conf = np.array(conf)
        acc = np.array(acc)
        bin_upper_bounds = np.linspace(0, 1, bins + 1)[1:]
        # split_idx = np.searchsorted(bin_upper_bounds, conf, 'left')
        split_idx = np.digitize(conf, bin_upper_bounds, right=True)
        data_len = len(split_idx)

        ece = np.zeros(bins)
        for i in range(bins):
            idx = split_idx == i
            if np.sum(idx) > 0:
                bin_avg_conf = np.mean(conf[idx])
                bin_avg_acc = np.mean(acc[idx])
                bin_prob = np.sum(idx) / data_len

                ece[i] = np.abs(bin_avg_acc - bin_avg_conf) * bin_prob

            # print(bin_avg_acc, bin_avg_conf, bin_prob, ece[i])

        eces.append(ece.sum())

    return np.mean(eces)


def group_data(data: tuple, label: List[int], cls: List[int]):
    # the idx should be output of np.unique(data), which is sorted.
    # assert len(set(idx)) == len(idx)
    tuple_num = len(data)
    data_group = [[[] for _ in cls] for _ in range(tuple_num)]
    for i, l in enumerate(label):
        for j in range(tuple_num):
            data_group[j][l].append(data[j][i])

    return data_group
