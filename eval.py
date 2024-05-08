import logging

import torch
from model.metrics import *
from model.label_shift_est import LSC
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from utilis.test_ft import test_ft
from model.model_init import model_init
from torch.nn import functional as F
from dataloader.Custom_Dataloader import FeatureDataset
import torch.nn as nn
import numpy as np
import importlib

model_paths = {
    'stage2':        'networks.stage_2',
    'none_cifar':    'networks.resnet_cifar',
    'none':          'networks.resnet',
    'tail_cifar':    'networks.resnet_cifar_ensemble',
    'tail':          'networks.resnet_ensemble'
}

def get_metrics(probs, labels, cls_num_list):
    labels = [tensor.cpu().item() for tensor in labels]
    acc = acc_cal(probs, labels, method='top1')

    mmf_acc = list(mmf_acc_cal(probs, labels, cls_num_list))
    logging.info('Many Medium Few shot Top1 Acc: ' + str(mmf_acc))
    print('Many Medium Few shot Top1 Acc: ' + str(mmf_acc))
    return acc, mmf_acc

# read from main.py directly: test_set, dset_info, dataset_info, args
def evaluation(test_set, dset_info, dataset_info, args, cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load the model for evaluate
    model = test_ft(datapath=dataset_info["path"],
                            args=args,
                            modelpath=args.eval,
                            crt_modelpath=None, test_cfg=None)
    model.to(device)
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    #### --------------- evaluate ---------------
    # Get number of batches
    num_batches = len(test_set)

    test_loss, correct, total = 0, 0, 0

    probs, labels = [], []

    # since we dont need to update the gradients, we use torch.no_grad()
    with torch.no_grad():
        for data in test_set:
            # Every data instance is an image + label pair
            img, label = data
            # Transfer data to target device
            img = img.to(device)
            label = label.to(device)
            labels.append(label)

            # Compute prediction for this batch
            logit = model(img)

            # compute the loss
            test_loss += loss_fn(logit, label).item()

            # Calculate the index of maximum logit as the predicted label
            prob = F.softmax(logit, dim=1)
            probs.extend(list(prob.squeeze().cpu().numpy()))
            pred = prob.argmax(dim=1)

            # record correct predictions
            correct += (pred == label).type(torch.float).sum().item()
            total += label.size(0)

    # -----------------Post Compensation Accuracy-------------------------------#
    probs = np.array(probs)
    labels = torch.cat(labels)
    _, mmf_acc = get_metrics(probs, labels, dset_info['per_class_img_num'])
    # Gather data and report
    test_loss /= num_batches
    accuracy = correct / total
    logging.info("Test Error:   Accuracy: {:.2f}, Avg loss: {:.4f} ".format(100 * accuracy, test_loss))
    print("Test Error:   Accuracy: {:.2f}, Avg loss: {:.4f} ".format(100 * accuracy, test_loss))


    pc_probs = LSC(probs, cls_num_list=dset_info['per_class_img_num'])
    label_shift_acc, mmf_acc_pc = get_metrics(pc_probs, labels, dset_info['per_class_img_num'])


    logging.info("Test Error:   Accuracy: {:.2f}, Avg loss: {:.4f} ".format(100 * accuracy, test_loss))
    print("Test Error:   Accuracy: {:.2f}, Avg loss: {:.4f} ".format(100 * accuracy, test_loss))

    logging.info("Label Shift Accracy is: {}".format(label_shift_acc))
    print("Label Shift Accracy is:", label_shift_acc)

    logging.info("\n")
    print("\n\n")
    return test_loss, accuracy, label_shift_acc, mmf_acc, mmf_acc_pc #FIXME