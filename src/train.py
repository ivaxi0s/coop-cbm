"""
Train InceptionV3 Network using the CUB-200-2011 dataset
"""
import pdb
import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_sel import *
import math
import torch
import torchvision as tv
import numpy as np
import torch.utils.data as td
from analysis import Logger, AverageMeter, accuracy, binary_accuracy
import sklearn.model_selection as skms
from src import probe, tti, gen_cub_spurious, hyperopt
from src.dataset import load_data, find_class_imbalance
from src.config import BASE_DIR, N_CLASSES, UPWEIGHT_RATIO, MIN_LR, LR_DECAY_SIZE, N_ATTRIBUTES
from src.models import ModelXtoCY, ModelXtoChat_ChatToY, ModelXtoY, ModelXtoC, ModelOracleCtoY, ModelXtoCtoY
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from src.col import ConceptOrthogonalLoss

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
    breakpoint()
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
                op_loss = ConceptOrthogonalLoss(lamb=args.lamb)
                if attr_criterion is not None and args.attr_col:
                    for i in range(args.n_attributes):
                        attr_op = op_loss(p_out, attr_labels_var[:,i])
                        losses.append(args.col_w * attr_op) 
            if not args.bottleneck: #loss main is for the main task label (always the first output)
                loss_main = 1.0 * criterion(outputs[0], labels_var) + 0.4 * criterion(aux_outputs[0], labels_var)
                losses.append(loss_main)
                out_start = 1
            if args.exp == 'Coop':
                loss_aux = 1.0 * criterion(outputs[1], labels_var) + 0.4 * criterion(aux_outputs[1], labels_var)
                losses.append(args.supp_w * loss_aux)
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

def train(model, args):

    trainset, validset, test_loader = selector(args)
    imbalance = None
    
    dir = os.path.join('outputfiles', args.exp, args.dset, str(args.n_attributes), str(args.col), str(args.attr_loss_weight))
    if not os.path.exists(dir):
        os.makedirs(dir)
    logger = Logger(os.path.join(dir, 'log.txt'))
    logger.write(str(args) + '\n')
    logger.write(str(imbalance) + '\n')
    logger.flush()
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    if args.use_attr and not args.no_img:
        attr_criterion = [] #separate criterion (loss function) for each attribute
        if args.weighted_loss:
            for i in range(args.n_attributes):
                attr_criterion.append(torch.nn.BCEWithLogitsLoss())
        else:
            for i in range(args.n_attributes):
                attr_criterion.append(torch.nn.CrossEntropyLoss())
    else:
        attr_criterion = None
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, threshold=0.00001, min_lr=0.00001, eps=1e-08)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=0.1)
    stop_epoch = int(math.log(MIN_LR / args.lr) / math.log(LR_DECAY_SIZE)) * args.scheduler_step
    print("Stop epoch: ", stop_epoch)

    # pdb.set_trace()
    if args.ckpt: #retraining
        train_loader = trainset
        val_loader = validset
    else:
        train_loader = trainset
        val_loader = validset
    
    best_val_epoch = -1
    best_val_loss = float('inf')
    best_val_acc = 0

    for epoch in range(0, args.epochs):
        train_loss_meter = AverageMeter()
        train_acc_meter = AverageMeter()
        if args.no_img:
            train_loss_meter, train_acc_meter = run_epoch_simple(model, optimizer, train_loader, train_loss_meter, train_acc_meter, criterion, args, is_training=True)
        else:
            train_loss_meter, train_acc_meter = run_epoch(model, optimizer, train_loader, train_loss_meter, train_acc_meter, criterion, attr_criterion, args, is_training=True)
 
        # if not args.ckpt: # evaluate on val set
        val_loss_meter = AverageMeter()
        val_acc_meter = AverageMeter()
    
        with torch.no_grad():
            if args.no_img:
                val_loss_meter, val_acc_meter = run_epoch_simple(model, optimizer, val_loader, val_loss_meter, val_acc_meter, criterion, args, is_training=False)
            else:
                val_loss_meter, val_acc_meter = run_epoch(model, optimizer, val_loader, val_loss_meter, val_acc_meter, criterion, attr_criterion, args, is_training=False)

        # else: #retraining
        #     val_loss_meter = train_loss_meter
        #     val_acc_meter = train_acc_meter

        if best_val_acc < val_acc_meter.avg:
            best_val_epoch = epoch
            best_val_acc = val_acc_meter.avg
            logger.write('New model best model at epoch %d\n' % epoch)
            torch.save(model, os.path.join(dir, 'best_model_%d.pth' % args.seed))
            #if best_val_acc >= 100: #in the case of retraining, stop when the model reaches 100% accuracy on both train + val sets
            #    break

        train_loss_avg = train_loss_meter.avg
        val_loss_avg = val_loss_meter.avg
        
        logger.write('Epoch [%d]:\tTrain loss: %.4f\tTrain accuracy: %.4f\t'
                'Val loss: %.4f\tVal acc: %.4f\t'
                'Best val epoch: %d\n'
                % (epoch, train_loss_avg, train_acc_meter.avg, val_loss_avg, val_acc_meter.avg, best_val_epoch)) 
        logger.flush()
        
        if epoch <= stop_epoch:
            scheduler.step() #scheduler step to update lr at the end of epoch     
        #inspect lr
        if epoch % 10 == 0:
            print('Current lr:', scheduler.get_last_lr())

        # if epoch % args.save_step == 0:
        #     torch.save(model, os.path.join(args.log_dir, '%d_model.pth' % epoch))

        if epoch >= 100 and val_acc_meter.avg < 3:
            print("Early stopping because of low accuracy")
            break
        if epoch - best_val_epoch >= 300:
            print("Early stopping because acc hasn't improved for a long time")
            break

def train_X_to_C(args):
    model = ModelXtoC(pretrained=args.pretrained, freeze=args.freeze, num_classes=N_CLASSES, use_aux=args.use_aux,
                      n_attributes=args.n_attributes, expand_dim=args.expand_dim, three_class=args.three_class)
    train(model, args)

def train_oracle_C_to_y_and_test_on_Chat(args):
    model = ModelOracleCtoY(n_class_attr=args.n_class_attr, n_attributes=args.n_attributes,
                            num_classes=N_CLASSES, expand_dim=args.expand_dim)
    train(model, args)

def train_Chat_to_y_and_test_on_Chat(args):
    model = ModelXtoChat_ChatToY(n_class_attr=args.n_class_attr, n_attributes=args.n_attributes,
                                 num_classes=N_CLASSES, expand_dim=args.expand_dim)
    train(model, args)

def train_X_to_C_to_y(args):
    if args.repeat_concepts:
        concepts_repeated = int(args.rep*args.n_attributes)
        args.n_attributes = args.n_attributes + concepts_repeated
    model = ModelXtoCtoY(n_class_attr=args.n_class_attr, pretrained=args.pretrained, freeze=args.freeze,
                         num_classes=N_CLASSES, use_aux=args.use_aux, n_attributes=args.n_attributes,
                         expand_dim=args.expand_dim, use_relu=args.use_relu, use_sigmoid=args.use_sigmoid)
    train(model, args)

def train_X_to_y(args):
    model = ModelXtoY(pretrained=args.pretrained, freeze=args.freeze, num_classes=N_CLASSES, use_aux=args.use_aux)
    train(model, args)

def train_X_to_Cy(args):
    if args.repeat_concepts:
        concepts_repeated = int(args.rep*args.n_attributes)
        args.n_attributes = args.n_attributes + concepts_repeated
    model = ModelXtoCY(pretrained=args.pretrained, freeze=args.freeze, num_classes=N_CLASSES, use_aux=args.use_aux,
                       n_attributes=args.n_attributes, three_class=args.three_class, connect_CY=args.connect_CY)
    train(model, args)

def train_probe(args):
    probe.run(args)

def test_time_intervention(args):
    tti.run(args)

def robustness(args):
    gen_cub_spurious.run(args)

def hyperparameter_optimization(args):
    hyperopt.run(args)


def parse_arguments(experiment):
    # Get argparse configs from user
    parser = argparse.ArgumentParser(description='CUB Training')
    parser.add_argument('exp', type=str,
                        choices=['Concept_XtoC', 'Independent_CtoY', 'Sequential_CtoY',
                                 'Standard', 'Coop', 'Joint', 'Probe',
                                 'TTI', 'Robustness', 'HyperparameterSearch'],
                        help='Name of experiment to run.')
    parser.add_argument('--seed', required=True, type=int, help='Numpy and torch seed.')

    if experiment == 'Probe':
        return (probe.parse_arguments(parser),)

    elif experiment == 'TTI':
        return (tti.parse_arguments(parser),)

    elif experiment == 'Robustness':
        return (gen_cub_spurious.parse_arguments(parser),)

    elif experiment == 'HyperparameterSearch':
        return (hyperopt.parse_arguments(parser),)

    else:
        parser.add_argument('-data_dir', default=None, help='where the trained model is saved')
        parser.add_argument('-log_dir', default=None, help='where the trained model is saved')
        parser.add_argument('-dset', default='birds', type = str, help='which dataset')
        parser.add_argument('-batch_size', '-b', type=int, help='mini-batch size')
        parser.add_argument('-epochs', '-e', type=int, help='epochs for training process')
        parser.add_argument('-save_step', default=1000, type=int, help='number of epochs to save model')
        parser.add_argument('-lr', type=float, help="learning rate")
        parser.add_argument('-weight_decay', type=float, default=5e-4, help='weight decay for optimizer')
        parser.add_argument('-pretrained', '-p', action='store_true',
                            help='whether to load pretrained model & just fine-tune')
        parser.add_argument('-freeze', action='store_true', help='whether to freeze the bottom part of inception network')
        parser.add_argument('-use_aux', action='store_true', help='whether to use aux logits')
        parser.add_argument('-use_attr', action='store_true',
                            help='whether to use attributes (FOR COTRAINING ARCHITECTURE ONLY)')
        parser.add_argument('-attr_loss_weight', default=1.0, type=float, help='weight for loss by predicting attributes')
        parser.add_argument('-no_img', action='store_true',
                            help='if included, only use attributes (and not raw imgs) for class prediction')
        parser.add_argument('-advattack', action='store_true',
                            help='if included, do an adv attack')
        parser.add_argument('-bottleneck', help='whether to predict attributes before class labels', action='store_true')
        parser.add_argument('-weighted_loss', default='', # note: may need to reduce lr
                            help='Whether to use weighted loss for single attribute or multiple ones')
        parser.add_argument('-uncertain_labels', action='store_true',
                            help='whether to use (normalized) attribute certainties as labels')
        parser.add_argument('-n_attributes', type=int, default=N_ATTRIBUTES,
                            help='whether to apply bottlenecks to only a few attributes')
        parser.add_argument('-expand_dim', type=int, default=0,
                            help='dimension of hidden layer (if we want to increase model capacity) - for bottleneck only')
        parser.add_argument('-n_class_attr', type=int, default=2,
                            help='whether attr prediction is a binary or triary classification')
        parser.add_argument('-resampling', help='Whether to use resampling', action='store_true')
        parser.add_argument('-end2end', action='store_true',
                            help='Whether to train X -> A -> Y end to end. Train cmd is the same as cotraining + this arg')
        parser.add_argument('-optimizer', default='SGD', help='Type of optimizer to use, options incl SGD, RMSProp, Adam')
        parser.add_argument('-ckpt', default='', help='For retraining on both train + val set')
        parser.add_argument('-scheduler_step', type=int, default=1000,
                            help='Number of steps before decaying current learning rate by half')
        parser.add_argument('-normalize_loss', action='store_true',
                            help='Whether to normalize loss by taking attr_loss_weight into account')
        parser.add_argument('-use_relu', action='store_true',
                            help='Whether to include relu activation before using attributes to predict Y. '
                                 'For end2end & bottleneck model')
        parser.add_argument('-use_sigmoid', action='store_true',
                            help='Whether to include sigmoid activation before using attributes to predict Y. '
                                 'For end2end & bottleneck model')
        parser.add_argument('-connect_CY', action='store_true',
                            help='Whether to use concepts as auxiliary features (in Coop) to predict Y')
        parser.add_argument('-col', action='store_true',
                            help='Whether to use othogonal projection loss')
        parser.add_argument('-attr_col', action='store_true',
                            help='Whether to use othogonal projection loss for attributes')
        parser.add_argument('-col_w', default=1.0, type=float, help='weight for col loss')
        parser.add_argument('-repeat_concepts', action='store_true', help = "whether you want concepts to be repeated")
        parser.add_argument('-rep', default=None, type=float, help=" percentage of concept repitition")
        parser.add_argument('-lamb', default=0.5, type=float, help="col loss weightage")
        parser.add_argument('-supp_w', default=0.01, type=float, help="loss weightage for supplementary multi label")
        parser.add_argument('-corruption_name', default='gaussian_blur', type=str, help="if you want to add image corruption")
        args = parser.parse_args()
        args.three_class = (args.n_class_attr == 3)
        return (args,)