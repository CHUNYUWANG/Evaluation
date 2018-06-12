import numpy as np


def evaluate(pred, gt):
    """
    pred: n * 14 * 3
    gt: n * 14 * 3
    """

    assert pred.shape == gt.shape
    errs = []
    for i in range(pred.shape[0]):
        pred_i = np.reshape(pred[i], 14, 3)
        gt_i = np.reshape(gt[i], 14, 3)
        mpje = np.mean(np.sum((pred_i-gt_i)**2, axis=1))
        errs.append(mpje)

    return np.mean(np.array(errs))