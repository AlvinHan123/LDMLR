import logging
from model.metrics import *
from model.label_shift_est import LSC

def metrics_cal(probs, labels, train_cls_num_list, ensemble_num=None):
    cls_num = len(train_cls_num_list)
    probs = np.array(probs)

    # ---------------------------Regular Accuracy-------------------------------#
    logging.info('#-----Normal Softmax Metric------#')
    normal_probs = probs.copy()
    if ensemble_num is not None:
        normal_probs = normal_probs.mean(-2)
    normal_metric = get_metrics(normal_probs, labels, train_cls_num_list)
    del normal_probs

    # -----------------Post Compensation Accuracy-------------------------------#
    logging.info('#-----Label Shift Compensation Softmax Metric------#')
    pc_probs = LSC(probs, cls_num_list=train_cls_num_list)
    if ensemble_num is not None:
        pc_probs = pc_probs.mean(-2)

    pc_metric = get_metrics(pc_probs, labels, train_cls_num_list)
    del pc_probs

    metrics = {'normal': normal_metric, 'pc': pc_metric}

    return normal_metric['acc'], metrics


def get_metrics(probs, labels, cls_num_list):
    acc = acc_cal(probs, labels, method='top1')
    logging.info('Evaluation Top1 Acc %.4f' % acc)

    mmf_acc = list(mmf_acc_cal(probs, labels, cls_num_list))
    logging.info('Many Medium Few shot Top1 Acc: ' + str(mmf_acc))

    # pc_ece = ece_loss(torch.Tensor(np.array(pc_probs)), torch.LongTensor(np.array(labels))).detach().cpu().numpy()
    ece = ECECal(np.array(probs), list(labels))
    sce = SCECal(np.array(probs), list(labels))
    bier = BierCal(np.array(probs), list(labels))
    ent = EntropyCal(np.array(probs))
    logging.info('ECE, SCE, Bier, Entropy of current model: %.4f, %.4f, %.4f, %.4f' % (ece, sce, bier, ent))

    result = {'acc': acc,
              'sce': sce,
              'ece': ece,
              'bier': bier,
              'entropy': ent,
              'mmf_acc': mmf_acc}

    result = round_val(result)

    return result


def round_val(metrics):
    # Round metric values to a more reader friendly form
    for k, v in metrics.items():
        if type(v) in [np.ndarray, list]:
            metrics[k] = [round(float(x), 4) for x in list(v)]
        else:
            metrics[k] = round(float(v), 4)

    return metrics
