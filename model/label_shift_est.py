import numpy as np
from typing import List
from utilis.utils import normalized



# Estimation of Target Classifier q(y|x)--------------------------------------#
def LSC(probs: np.ndarray, cls_num_list: List):
    r"""
    Implementation of Label Shift Compensation (LSC) with known target label distribution.
    Given source domain p(y=c_i) and p(y|x), target domain q(y=c_i),
    estimate target predicted probability q(y|x) on test set.

    Args:
        probs:          Softmax probability p(y|x) predicted by the classifier,
                        for all samples in test set (N x C).
        cls_num_list:   Number of sample for each class in train set.

    Shapes:
        * Input:
            probs:              N x C   (No. of samples) x (No. of classes),
            cls_num_list:       C       (No. of classes),
        * Output:
            pc_probs:           N x C   (No. of samples) x (No. of classes)


    For more information see paper:
    [2002] "Adjusting the Outputs of a Classifier to New a Priori Probabilities: A Simple Procedure"
    """
    cls_num = len(cls_num_list)
    py = np.array(cls_num_list) / cls_num
    qy = np.ones(cls_num) / cls_num

    w = qy / py
    pc_probs = normalized(probs * w, axis=-1, order=1)

    return pc_probs
