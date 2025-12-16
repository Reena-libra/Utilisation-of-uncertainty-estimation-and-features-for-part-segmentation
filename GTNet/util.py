#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: util
@Time: 4/5/19 3:47 PM
"""

import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F


def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        #one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1).long(), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss



def cal_loss_new(pred, gold, weights=None,batch_size=None,num_points=None, smoothing=True):
        #print('pred',pred.shape,gold.shape)
        gold = gold.contiguous().view(-1).long()  # Flatten target labels for calculation
       
        if smoothing:
            eps = 0.2
            n_class = pred.size(1)
            #print('gold',gold.shape)
            # One-hot encoding for label smoothing
            one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
            #print(one_hot.shape)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            # Calculate per-sample loss with label smoothing
            per_sample_loss = -(one_hot * log_prb).sum(dim=1)
            #print(per_sample_loss.shape)
            #per_sample_loss = -(one_hot * log_prb).sum(dim=2)
        else:
            # Standard cross-entropy without label smoothing
            per_sample_loss = F.cross_entropy(pred, gold, reduction='none')  # per-sample loss
            #per_sample_loss = per_sample_loss.view(batch_size, num_points)  # Reshape to [batch_size, num_points]
        
        per_sample_loss = per_sample_loss.view(batch_size, num_points)  # Shape: [batch_size, num_points]
        
        # Average over points within each sample to get per-sample loss
        per_sample_loss = per_sample_loss.mean(dim=1)  # Shape: [batch_size]
        #per_sample_loss = per_point_loss.mean(dim=1)
        # Apply weights if provided
        if weights is not None:
            #print('persample',per_sample_loss.shape)
            #per_sample_loss=per_sample_loss.view(num_points,batch_size)
            #print('loss',per_sample_loss.shape, weights.shape)
            per_sample_loss = per_sample_loss * weights  # Element-wise multiplication with weights

        # Return mean of weighted losses
        return per_sample_loss.mean()



def mc_dropout_variance(model,  inputs, cls_label, num_samples=5):
        #model.train()  # Ensure dropout is active during inference
        predictions = []
        
        with torch.no_grad():
            for i in range(num_samples):
                # Perform forward pass with inputs and class labels
                seg_pred = model(inputs, cls_label)  # seg_pred: [batch_size, num_points, num_classes]
                predictions.append(seg_pred.unsqueeze(0))  # Add a new dimension for stacking
        
        # Stack predictions across num_samples
        predictions = torch.cat(predictions, dim=0)  # Shape: [num_samples, batch_size, num_points, num_classes]

        # Calculate mean and variance across MC-Dropout samples (across num_samples dimension)
        variance_pred = predictions.var(dim=0)  # Variance of predictions: [batch_size, num_points, num_classes]

        # Calculate the mean variance for each sample by averaging over points and classes
        mean_variance_per_sample = variance_pred.mean(dim=[1, 2])  # Shape: [batch_size]
        
        return mean_variance_per_sample
    


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush() #flush() 方法是用来刷新缓冲区的

    def close(self):
        self.f.close()

import os
import numpy as np
from PIL import Image

import torch
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.batchnorm import _BatchNorm
import torch.nn.init as initer

def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    # https://github.com/pytorch/pytorch/issues/1382
    area_intersection = torch.histc(intersection.float().cpu(), bins=K, min=0, max=K-1)
    area_output = torch.histc(output.float().cpu(), bins=K, min=0, max=K-1)
    area_target = torch.histc(target.float().cpu(), bins=K, min=0, max=K-1)
    area_union = area_output + area_target - area_intersection
    return area_intersection.cuda(), area_union.cuda(), area_target.cuda()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def step_learning_rate(optimizer, base_lr, epoch, step_epoch, multiplier=0.1, clip=1e-6):
    """Sets the learning rate to the base LR decayed by 10 every step epochs"""
    lr = max(base_lr * (multiplier ** (epoch // step_epoch)), clip)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def poly_learning_rate(optimizer, base_lr, curr_iter, max_iter, power=0.9):
    """poly learning rate policy"""
    lr = base_lr * (1 - float(curr_iter) / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = 255
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))
    area_output, _ = np.histogram(output, bins=np.arange(K+1))
    area_target, _ = np.histogram(target, bins=np.arange(K+1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    # https://github.com/pytorch/pytorch/issues/1382
    area_intersection = torch.histc(intersection.float().cpu(), bins=K, min=0, max=K-1)
    area_output = torch.histc(output.float().cpu(), bins=K, min=0, max=K-1)
    area_target = torch.histc(target.float().cpu(), bins=K, min=0, max=K-1)
    area_union = area_output + area_target - area_intersection
    return area_intersection.cuda(), area_union.cuda(), area_target.cuda()


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def check_makedirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def init_weights(model, conv='kaiming', batchnorm='normal', linear='kaiming', lstm='kaiming'):
    """
    :param model: Pytorch Model which is nn.Module
    :param conv:  'kaiming' or 'xavier'
    :param batchnorm: 'normal' or 'constant'
    :param linear: 'kaiming' or 'xavier'
    :param lstm: 'kaiming' or 'xavier'
    """
    for m in model.modules():
        if isinstance(m, (_ConvNd)):
            if conv == 'kaiming':
                initer.kaiming_normal_(m.weight)
            elif conv == 'xavier':
                initer.xavier_normal_(m.weight)
            else:
                raise ValueError("init type of conv error.\n")
            if m.bias is not None:
                initer.constant_(m.bias, 0)

        elif isinstance(m, _BatchNorm):
            if batchnorm == 'normal':
                initer.normal_(m.weight, 1.0, 0.02)
            elif batchnorm == 'constant':
                initer.constant_(m.weight, 1.0)
            else:
                raise ValueError("init type of batchnorm error.\n")
            initer.constant_(m.bias, 0.0)

        elif isinstance(m, nn.Linear):
            if linear == 'kaiming':
                initer.kaiming_normal_(m.weight)
            elif linear == 'xavier':
                initer.xavier_normal_(m.weight)
            else:
                raise ValueError("init type of linear error.\n")
            if m.bias is not None:
                initer.constant_(m.bias, 0)

        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    if lstm == 'kaiming':
                        initer.kaiming_normal_(param)
                    elif lstm == 'xavier':
                        initer.xavier_normal_(param)
                    else:
                        raise ValueError("init type of lstm error.\n")
                elif 'bias' in name:
                    initer.constant_(param, 0)





def colorize(gray, palette):
    # gray: numpy array of the label and 1*3N size list palette
    color = Image.fromarray(gray.astype(np.uint8)).convert('P')
    color.putpalette(palette)
    return color


class calculate_density(nn.Module):
    def __init__(self):
        super(calculate_density, self).__init__()

    def forward(self, labels_batch):
        """
        Compute density-based weights based on the number of 'ear' points (label == 1) in each sample.

        Args:
            labels_batch: Tensor of shape [B, N] where B is batch size and N is number of points

        Returns:
            class_weights: Tensor of shape [B] with weights per sample
        """
        batch_size = labels_batch.size(0)
        class_weights = []

        for i in range(batch_size):
            labels = labels_batch[i, :]  # Shape: [N]
           # print('labels',labels.shape)
            num_ear_points = torch.sum(labels == 1)
            total_points = labels.numel()
            #print('total',total_points,num_ear_points)

            density_ratio = torch.true_divide(num_ear_points, total_points)  # scalar
            cls_weight = 1.0 / (1.0 + torch.exp(-density_ratio))  # Apply sigmoid
            class_weights.append(cls_weight)

        return torch.stack(class_weights)  # Return as Tensor of shape [B]