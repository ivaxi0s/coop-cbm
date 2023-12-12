
import pdb
import sys, os

if not os.path.isdir('CMDs'):
    os.mkdir('CMDs')
with open('CMDs/experiments.cmd', 'a') as f:
    f.write(' '.join(sys.argv)+'\n')

def run_experiments(args):
    from src.train import (
        train_X_to_C,
        train_oracle_C_to_y_and_test_on_Chat,
        train_Chat_to_y_and_test_on_Chat,
        train_X_to_C_to_y,
        train_X_to_y,
        train_X_to_Cy,
        train_probe,
        test_time_intervention,
        robustness,
        hyperparameter_optimization
    )

    experiment = args[0].exp
    if experiment == 'Concept_XtoC':
        train_X_to_C(*args)

    elif experiment == 'Independent_CtoY':
        train_oracle_C_to_y_and_test_on_Chat(*args)

    elif experiment == 'Sequential_CtoY':
        train_Chat_to_y_and_test_on_Chat(*args)

    elif experiment == 'Joint':
        train_X_to_C_to_y(*args)

    elif experiment == 'Standard':
        train_X_to_y(*args)

    elif experiment == 'StandardWithAuxC':
        train_X_to_y_with_aux_C(*args)

    elif experiment == 'Coop':
        train_X_to_Cy(*args)

    elif experiment == 'Probe':
        train_probe(*args)

    elif experiment == 'TTI':
        test_time_intervention(*args)

    elif experiment == 'Robustness':
        robustness(*args)

    elif experiment == 'HyperparameterSearch':
        hyperparameter_optimization(*args)

def parse_arguments():
    # First arg must be dataset, and based on which dataset it is, we will parse arguments accordingly
    assert len(sys.argv) > 2, 'You need to specify dataset and experiment'
    assert sys.argv[1] in ['Concept_XtoC', 'Independent_CtoY', 'Sequential_CtoY',
                           'Standard', 'StandardWithAuxC', 'Coop', 'Joint', 'Probe',
                           'TTI', 'Robustness', 'HyperparameterSearch'], \
        'Please specify valid experiment. Current: %s' % sys.argv[2]
    experiment = sys.argv[1].upper()

    # Handle accordingly to dataset
    from src.train import parse_arguments

    args = parse_arguments(experiment=experiment)
    return  args

if __name__ == '__main__':

    import torch
    import numpy as np

    args = parse_arguments()

    # Seeds
    np.random.seed(args[0].seed)
    torch.manual_seed(args[0].seed)

    run_experiments(args)
