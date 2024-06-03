# code reference: https://colab.research.google.com/drive/1IJkrrV-D7boSCLVKhi7t5docRYqORtm3#scrollTo=jpy3GC7XzC7J
import os
import argparse
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import time
import logging
import random
import math
from copy import deepcopy
from contextlib import contextmanager
from collections import defaultdict

import torch
from torch import optim, nn
from torch.nn import functional as F
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from torch.utils.data import TensorDataset, DataLoader

from utilis.utils import move_model_to_cpu
from utilis.test_ft import test_ft
from utilis.metric import cal_FID_per_class, calculate_fid, unload_feature_vectors
from model.ddpm_conditional import UNet_conditional, ConditionalDiffusion1D


def generate_class_specific_noise(class_num, channel_num = 1, sequence_length = 64):
    torch.manual_seed(123)
    class_specific_noises = []

    for _ in range(class_num):
        # Generate random noise with shape (num_batch, 1, noise_dim)
        noise = torch.randn(1, channel_num, sequence_length)
        class_specific_noises.append(noise)
    return class_specific_noises


def label_to_noise(labels, class_num, variance=0.001):
    class_specific_noise = generate_class_specific_noise(class_num)
    class_noise = []
    for label in labels:
        mean = class_specific_noise[label]
        # Sample from a normal distribution with the given mean and variance
        noise = torch.normal(mean, variance)
        class_noise.append(noise)

    class_noise = torch.cat(class_noise, dim=0)
    return class_noise


# exponential moving average to smooth the loss
@torch.no_grad()
def ema_update(model, averaged_model, decay):
    """Incorporates updated model parameters into an exponential moving averaged
    version of a model. It should be called after each optimizer step."""
    model_params = dict(model.named_parameters())
    averaged_params = dict(averaged_model.named_parameters())
    assert model_params.keys() == averaged_params.keys()

    for name, param in model_params.items():
        averaged_params[name].mul_(decay).add_(param, alpha=1 - decay)

    model_buffers = dict(model.named_buffers())
    averaged_buffers = dict(averaged_model.named_buffers())
    assert model_buffers.keys() == averaged_buffers.keys()

    for name, buf in model_buffers.items():
        averaged_buffers[name].copy_(buf)


def diffusion_train(dataloader_train, dataloader_test, feature_dataloader_tr, dataset_info, dset_info, args):
    # Create the model and optimizer
    seed = 123
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = torch.cuda.amp.GradScaler()
    print('Using device:', device)
    torch.manual_seed(123)

    if args.dataset == "CIFAR10" or args.dataset == "CIFAR100":
        model = UNet_conditional(
            dim=64,
            dim_mults=(1, 2, 4, 8),
            channels=1,
            num_classes=dset_info["class_num"]
        )

        diffusion_model = ConditionalDiffusion1D(
            model,
            seq_length=64,
            timesteps=1000,
            objective='pred_x0'
        )
        
    elif args.dataset == "ImageNet":
        model = UNet_conditional(
            dim=512,
            dim_mults=(1, 2, 4),
            channels=1,
            num_classes=dset_info["class_num"]
        )

        diffusion_model = ConditionalDiffusion1D(
            model,
            seq_length=512,
            timesteps=1000,
            objective='pred_x0'
        )

    print(diffusion_model,"\n\n\n",model)

    # Calculate the number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    # If you want to count only the trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {trainable_params}")

    diffusion_model.to(device)

    if args.is_diffusion_pretrained:
        diffusion_model = torch.load(args.is_diffusion_pretrained)
        print("pre-trained loaded!")
    else:
        optimizer_diffusion = optim.SGD(diffusion_model.parameters(), lr=0.0001)

        # --------------------- train loop ---------------------
        for epoch in range(args.diffusion_epoch):
            total_diffusion_loss = 0.0

            for batch, data in enumerate(feature_dataloader_tr):
                feature, feature_lable = data
                feature  = feature.unsqueeze(1)
                feature = feature.to(device)
                feature_lable = feature_lable.to(device)

                optimizer_diffusion.zero_grad()
                diffusion_loss = diffusion_model(feature, feature_lable)
                total_diffusion_loss += diffusion_loss.item()

                # Backpropagation
                # Do the optimizer step and EMA update
                scaler.scale(diffusion_loss).backward()
                scaler.step(optimizer_diffusion)
                scaler.update()
            print("epoch: {}, the diffusion loss is {}".format(epoch, (total_diffusion_loss / (batch+1))))
            logging.info("epoch: {}, the diffusion loss is {}".format(epoch, (total_diffusion_loss / (batch+1))))

            if epoch % 50 == 0:
                # Ensure the directory exists
                directory = 'pretrained_models'
                if not os.path.exists(directory):
                    os.makedirs(directory)

                # Format the file name more cleanly
                imb_factor_formatted = int(args.imb_factor * 100)
                filename = 'diffusion_model_{}_imb_{}_epoch_{}.pt'.format(args.dataset, imb_factor_formatted, epoch)

                # Complete path
                file_path = os.path.join(directory, filename)

                # Save the model
                torch.save(diffusion_model, file_path)

    # --------------------- sample features ---------------------
    if args.dataset == "CIFAR10":
        class_num = []
        for index, num in enumerate(dset_info['per_class_img_num']):
            if args.generation_mmf == "many":
                if index >=0 and index < 4:
                    class_num.append(max(round(num * args.feature_ratio), 5))
                else:
                    class_num.append(0)
            elif args.generation_mmf == "medium":
                if index >=4 and index < 7:
                    class_num.append(max(round(num * args.feature_ratio), 5))
                else:
                    class_num.append(0)
            elif args.generation_mmf == "few":
                if index >= 7 and index < 10:
                    class_num.append(max(round(num * args.feature_ratio), 5))
                else:
                    class_num.append(0)
            else:
                class_num.append(max(round(num * args.feature_ratio), 5))
    elif args.dataset == "CIFAR100":
        class_num = []
        for index, num in enumerate(dset_info['per_class_img_num']):
            if args.generation_mmf == "many":
                if index >=0 and index < 34:
                    class_num.append(max(round(num * args.feature_ratio), 5))
                else:
                    class_num.append(0)
            elif args.generation_mmf == "medium":
                if index >=34 and index < 67:
                    class_num.append(max(round(num * args.feature_ratio), 5))
                else:
                    class_num.append(0)
            elif args.generation_mmf == "few":
                if index >= 67 and index < 100:
                    class_num.append(max(round(num * args.feature_ratio), 5))
                else:
                    class_num.append(0)
            else:
                class_num.append(max(round(num * args.feature_ratio), 5))
    elif args.dataset == "ImageNet":
        class_num = []
        for index, num in enumerate(dset_info['per_class_img_num']):
            if args.generation_mmf == "many":
                if num >= 100:
                    class_num.append(max(round(num * args.feature_ratio), 5))
                else:
                    class_num.append(0)
            elif args.generation_mmf == "few":
                if num <= 20:
                    class_num.append(max(round(num * args.feature_ratio), 5))
                else:
                    class_num.append(0)
            elif args.generation_mmf == "medium":
                if num < 100 and num > 20:
                    class_num.append(max(round(num * args.feature_ratio), 5))
                else:
                    class_num.append(0)
            else:
                class_num.append(max(round(num * args.feature_ratio), 5))
                
    # FIXME sample features by diffusion seperately
    # Move model to CPU and clear CUDA cache
    move_model_to_cpu(diffusion_model)
    torch.cuda.empty_cache()

    # Generate fake_classes
    fake_classes = [i for i, count in enumerate(class_num) for _ in range(count)]
    print("the number of generated features is ",len(fake_classes))
    logging.info("the number of generated features is {}".format(len(fake_classes)))
    random.shuffle(fake_classes)

    # Split fake_classes into smaller batches
    batch_size = 1024  # Adjust this based on your memory constraints
    fake_classes_batches = [fake_classes[i:i + batch_size] for i in range(0, len(fake_classes), batch_size)]

    # Initialize lists to store results
    generated_features_list = []
    fake_classes_list = []

    # Process each batch
    for batch in fake_classes_batches:
        logging.info("sampling batch:" + str(len(batch)))
        batch_tensor = torch.tensor(batch).to(device)
        diffusion_model.to(device)

        # Perform sampling for the current batch
        batch_generated_features = diffusion_model.sample(batch_tensor)

        # Move results to CPU and add to lists
        batch_generated_features = batch_generated_features.cpu()
        generated_features_list.append(batch_generated_features)
        fake_classes_list.append(batch_tensor.cpu())

        # Clear CUDA cache to free up memory
        torch.cuda.empty_cache()

    # Concatenate all results
    generated_features = torch.cat(generated_features_list, dim=0)
    fake_classes = torch.cat(fake_classes_list, dim=0)

    # Return the concatenated results
    return generated_features, fake_classes
