import logging

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset

from model.metrics import *
from model.label_shift_est import LSC

from utilis.test_ft import test_ft
from dataloader.Custom_Dataloader import FeatureDataset


# @title Evaluation Loop
def evaluate_loop(dataloader, model, loss_fn, device, dset_info):
    # Get number of batches
    num_batches = len(dataloader)
    test_loss, correct, total = 0, 0, 0
    probs, labels = [], []

    # since we dont need to update the gradients, we use torch.no_grad()
    with torch.no_grad():
        for data in dataloader:
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


def get_metrics(probs, labels, cls_num_list):
    labels = [tensor.cpu().item() for tensor in labels]
    acc = acc_cal(probs, labels, method='top1')

    mmf_acc = list(mmf_acc_cal(probs, labels, cls_num_list))
    logging.info('Many Medium Few shot Top1 Acc: ' + str(mmf_acc))
    print('Many Medium Few shot Top1 Acc: ' + str(mmf_acc))
    return acc, mmf_acc #FIXME


def fine_tune_fc(generated_features, fake_classes, feature_dataset_tr, test_set, dataset_info, args, dset_info):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move the data to the specified device
    generated_features = generated_features.squeeze(1).to(device)
    fake_classes = fake_classes.to(device)

    # Create a new TensorDataset
    generated_dataset = TensorDataset(generated_features, fake_classes)

    # Before merging datasets, ensure that feature_dataset_tr is also on the same device
    # Assuming feature_dataset_tr is a tuple containing features and labels
    features, labels = feature_dataset_tr.tensors
    features, labels = features.to(device), labels.to(device)
    feature_dataset_tr = TensorDataset(features, labels)

    # Merge datasets
    combined_dataset = ConcatDataset([generated_dataset, feature_dataset_tr])

    # Create DataLoader
    data_loader = DataLoader(combined_dataset, batch_size=args.batch_size_fc, shuffle=True)

    loss_fn = nn.CrossEntropyLoss()

    # load the model for fine tuning
    model_totrain = test_ft(datapath=dataset_info["path"],
                            args=args,
                            modelpath=args.model_fixed,
                            crt_modelpath=None, test_cfg=None)
    model_totrain.to(device)

    # First, freeze all layers and unfreeze the fully connected layer
    for param in model_totrain.parameters():
        param.requires_grad = False
    for param in model_totrain.fc.parameters():
        param.requires_grad = True

    best_accuracy, best_label_shift_acc = 0.0, 0.0
    best_mmf_acc_ce, best_mmf_acc_pc = [], []
    optimizer = torch.optim.Adam(model_totrain.parameters(), lr=args.learning_rate_fc, weight_decay=0.00001)
    for epoch in range(args.epoch):
        model_totrain.train()
        running_loss = 0.0
        total_loss = 0.0
        correct, total = 0, 0

        for batch, data in enumerate(data_loader):
            features, classes = data
            features, classes = features.to(device), classes.to(device)

            optimizer.zero_grad()

            logits = model_totrain.fc(features)
            probs = F.softmax(logits, dim=1)
            predition = probs.argmax(dim=1)

            # record correct predictions
            correct += (predition == classes).type(torch.float).sum().item()
            total += classes.size(0)

            # compute the loss and its gradients
            loss = loss_fn(logits, classes)
            # Backpropagation
            loss.backward()

            # update the parameters according to gradients
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            total_loss += loss.item()

        cnn_running_loss, train_accuray = total_loss / (batch+1), correct / total
        logging.info("cnn training loss is {}; cnn training accuracy is {:.2f}".format(cnn_running_loss, train_accuray*100))
        print("cnn training loss is {}; cnn training accuracy is {:.2f}".format(cnn_running_loss, train_accuray*100))

        # evaluate the model
        model_totrain.eval()
        print("epoch: {}: ".format(epoch))
        logging.info("epoch: {}: ".format(epoch))
        valid_loss, valid_accuracy, label_shift_acc, mmf_acc, mmf_acc_pc  = evaluate_loop(test_set, model_totrain, loss_fn, device, dset_info)# fixme
        if valid_accuracy > best_accuracy:  # save the model with best validation accuracy
            best_accuracy = valid_accuracy
            best_mmf_acc_ce = mmf_acc # fixme
        if label_shift_acc > best_label_shift_acc:
            best_label_shift_acc = label_shift_acc
            best_mmf_acc_pc = mmf_acc_pc # fixme
    logging.info("The best accuracy is {}, The best label shift accuracy is {}".format(best_accuracy, best_label_shift_acc))
    logging.info("the best accuracy ce mmf is {}; the best acc pc mmf is {}".format(best_mmf_acc_ce, best_mmf_acc_pc)) # fixme
    print("The best accuracy is {}, The best label shift accuracy is {}".format(best_accuracy, best_label_shift_acc))
