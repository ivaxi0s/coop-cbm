"""
Common functions for visualization in different ipython notebooks
"""
import os
import random
import argparse
import torch
import numpy as np

from matplotlib.pyplot import figure, imshow, axis, show
from matplotlib.image import imread

N_CLASSES = 200
N_ATTRIBUTES = 312

def get_class_attribute_names(img_dir = 'CUB_200_2011/images/', feature_file='CUB_200_2011/attributes/attributes.txt'):
    """
    Returns:
    class_to_folder: map class id (0 to 199) to the path to the corresponding image folder (containing actual class names)
    attr_id_to_name: map attribute id (0 to 311) to actual attribute name read from feature_file argument
    """
    class_to_folder = dict()
    for folder in os.listdir(img_dir):
        class_id = int(folder.split('.')[0])
        class_to_folder[class_id - 1] = os.path.join(img_dir, folder)

    attr_id_to_name = dict()
    with open(feature_file, 'r') as f:
        for line in f:
            idx, name = line.strip().split(' ')
            attr_id_to_name[int(idx) - 1] = name
    return class_to_folder, attr_id_to_name

def sample_files(class_label, class_to_folder, number_of_files=10):
    """
    Given a class id, extract the path to the corresponding image folder and sample number_of_files randomly from that folder
    """
    folder = class_to_folder[class_label]
    class_files = random.sample(os.listdir(folder), number_of_files)
    class_files = [os.path.join(folder, f) for f in class_files]
    return class_files

def show_img_horizontally(list_of_files):
    """
    Given a list of files, display them horizontally in the notebook output
    """
    fig = figure(figsize=(40,40))
    number_of_files = len(list_of_files)
    for i in range(number_of_files):
        a=fig.add_subplot(1,number_of_files,i+1)
        image = imread(list_of_files[i])
        imshow(image)
        axis('off')
    show(block=True)

def set_seed(seed_value=42):
    np.random.seed(seed_value)  
    torch.manual_seed(seed_value)  
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)  
        torch.cuda.manual_seed_all(seed_value)  
        torch.backends.cudnn.deterministic = True  
        torch.backends.cudnn.benchmark = False
    
    random.seed(seed_value)  
    os.environ['PYTHONHASHSEED'] = str(seed_value)

def parse_arguments(experiment):
    # Get argparse configs from user
    parser = argparse.ArgumentParser(description='CUB Training')
    parser.add_argument('dataset', type=str, help='Name of the dataset.')
    parser.add_argument('exp', type=str,
                        choices=['Concept_XtoC', 'Independent_CtoY', 'Sequential_CtoY',
                                 'Standard', 'Coop', 'Joint', 'Probe',
                                 'TTI', 'Robustness', 'HyperparameterSearch'],
                        help='Name of experiment to run.')
    parser.add_argument('--seed', required=True, type=int, help='Numpy and torch seed.')
    set_seed(args.seed)

    if experiment == 'Probe':
        return (probe.parse_arguments(parser),)

    elif experiment == 'TTI':
        return (tti.parse_arguments(parser),)

    elif experiment == 'Robustness':
        return (gen_spurious.parse_arguments(parser),)

    elif experiment == 'HyperparameterSearch':
        return (hyperopt.parse_arguments(parser),)

    else:
        parser.add_argument('-log_dir', default=None, help='where the trained model is saved')
        parser.add_argument('-data_dir', default=None, help='where the dataset is saved')
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
        parser.add_argument('-gamma', default=0.5, type=float, help="col loss weightage")
        parser.add_argument('-corruption_name', default='gaussian_blur', type=str, help="if you want to add image corruption")
        args = parser.parse_args()
        args.three_class = (args.n_class_attr == 3)
        return (args,)