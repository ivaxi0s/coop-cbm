import os
import sys
import torch
import torch.utils.data as td
import numpy as np
import torchvision as tv
import sklearn.model_selection as skms
from torch.autograd import Variable

from src.model import probe, hyperopt, models
from src.eval import tti
from src.data.old_dataset import load_data, find_class_imbalance
from src.util.config import BASE_DIR, N_CLASSES, UPWEIGHT_RATIO, MIN_LR, LR_DECAY_SIZE, N_ATTRIBUTES
from analysis import Logger, AverageMeter, accuracy, binary_accuracy
from src.col import ConceptOrthogonalLoss

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def run_epoch_simple(model, optimizer, loader, loss_meter, acc_meter, criterion, args, is_training):
    """
    A -> Y: Predicting class labels using only attributes with MLP
    """
    if is_training:
        model.train()
    else:
        model.eval()
    for _, data in enumerate(loader):
        inputs, labels = data
        if isinstance(inputs, list):
            #inputs = [i.long() for i in inputs]
            inputs = torch.stack(inputs).t().float()
        inputs = torch.flatten(inputs, start_dim=1).float()
        inputs_var = torch.autograd.Variable(inputs).to(device)
        inputs_var = inputs_var.to(device) if torch.cuda.is_available() else inputs_var
        labels_var = torch.autograd.Variable(labels).to(device)
        labels_var = labels_var.to(device) if torch.cuda.is_available() else labels_var
        
        outputs = model(inputs_var)
        loss = criterion(outputs, labels_var)
        acc = accuracy(outputs, labels, topk=(1,))
        loss_meter.update(loss.item(), inputs.size(0))
        acc_meter.update(acc[0], inputs.size(0))

        if is_training:
            optimizer.zero_grad() #zero the parameter gradients
            loss.backward()
            optimizer.step() #optimizer step to update parameters
    return loss_meter, acc_meter

def run_epoch(model, optimizer, loader, loss_meter, acc_meter, criterion, attr_criterion, args, is_training):
    """
    For the rest of the networks (X -> A, cotraining, simple finetune)
    """
    if is_training:
        model.train()
    else:
        model.eval()
    for _, data in enumerate(loader):
        if attr_criterion is None:
            inputs, labels = data
            attr_labels, attr_labels_var = None, None
        else:
            inputs, labels, attr_labels = data
            attr_labels = torch.tensor(attr_labels)
            labels = torch.tensor(labels)
            if args.n_attributes > 1:
                attr_labels = [i.long() for i in attr_labels]
                attr_labels = torch.stack(attr_labels).float() #N x 312
            else:
                if isinstance(attr_labels, list):
                    attr_labels = attr_labels[0]
                attr_labels = attr_labels.unsqueeze(1)
            attr_labels_var = torch.autograd.Variable(attr_labels).float()
            attr_labels_var = attr_labels_var.to(device) if torch.cuda.is_available() else attr_labels_var
        inputs_var = torch.autograd.Variable(inputs)
        inputs_var = inputs_var.to(device) if torch.cuda.is_available() else inputs_var

        labels_var = torch.autograd.Variable(labels)
        labels_var = labels_var.to(device) if torch.cuda.is_available() else labels_var
        if is_training and args.use_aux:
            p_out, outputs, aux_outputs = model(inputs_var)
            losses = []
            out_start = 0
            if args.col:
                op_loss = ConceptOrthogonalLoss(gamma=args.gamma)
                loss_op = op_loss(p_out,labels_var)
                losses.append(args.col_w * loss_op)
                # pdb.set_trace()
                if attr_criterion is not None and args.attr_col:
                    for i in range(args.n_attributes):
                        attr_op = op_loss(p_out, attr_labels_var[:,i])
                        losses.append(0.001 * attr_op) #############---------name this as lambda or something
            if not args.bottleneck: #loss main is for the main task label (always the first output)
                loss_main = 1.0 * criterion(outputs[0], labels_var) + 0.4 * criterion(aux_outputs[0], labels_var)
                losses.append(loss_main)
                out_start = 1
            if args.exp == 'Coop':
                loss_aux = 1.0 * criterion(outputs[1], labels_var) + 0.4 * criterion(aux_outputs[1], labels_var)
                losses.append(loss_aux)
                out_start = 2
            if attr_criterion is not None and args.attr_loss_weight > 0: #X -> A, cotraining, end2end
                
                for i in range(args.n_attributes):
                    
                    losses.append(args.attr_loss_weight * (1.0 * attr_criterion[i](outputs[i+out_start].squeeze().type(torch.FloatTensor).to(device), attr_labels_var[:, i]) \
                                                            + 0.4 * attr_criterion[i](aux_outputs[i+out_start].squeeze().type(torch.FloatTensor).to(device), attr_labels_var[:, i])))
        else: #testing or no aux logits
            outputs = model(inputs_var)
            losses = []
            out_start = 0
            if not args.bottleneck:
                loss_main = criterion(outputs[0], labels_var)
                losses.append(loss_main)
                out_start = 1
            if args.exp == 'Coop':
                loss_aux = criterion(outputs[1], labels_var) 
                losses.append(loss_aux)
                out_start = 2
            if attr_criterion is not None and args.attr_loss_weight > 0: #X -> A, cotraining, end2end
                for i in range(len(attr_criterion)):
                    losses.append(args.attr_loss_weight * attr_criterion[i](outputs[i+out_start].squeeze().type(torch.FloatTensor).to(device), attr_labels_var[:, i]))

        if args.bottleneck: #attribute accuracy
            sigmoid_outputs = torch.nn.Sigmoid()(torch.cat(outputs, dim=1))
            acc = binary_accuracy(sigmoid_outputs, attr_labels)
            acc_meter.update(acc.data.cpu().numpy(), inputs.size(0))
        else:
            acc = accuracy(outputs[0], labels, topk=(1,)) #only care about class prediction accuracy
            acc_meter.update(acc[0], inputs.size(0))

        if attr_criterion is not None:
            if args.bottleneck:
                total_loss = sum(losses)/ args.n_attributes
            else: #cotraining, loss by class prediction and loss by attribute prediction have the same weight
                total_loss = losses[0] + sum(losses[1:])
                if args.normalize_loss:
                    total_loss = total_loss / (1 + args.attr_loss_weight * args.n_attributes)
        else: #finetune
            total_loss = sum(losses)
        loss_meter.update(total_loss.item(), inputs.size(0))
        if is_training:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
    
    return loss_meter, acc_meter
