# coop-cbm
Repository for Neurips 2023 paper - Auxiliary Losses for Learning Generalizable Concept-based Model


# Auxiliary Losses for Learning Generalizable Concept-based Model

[Ivaxi Sheth](https://ivaxi0s.github.io/), [Samira Ebrahimi Kahou](https://saebrahimi.github.io/)

This repo provides code for our Neurips 2023 paper - Auxiliary Losses for Learning Generalizable Concept-based Model. 

To run the models for coop-CBM with COL, run the following command:
``` experiments.py Coop --seed=1 -ckpt 1 -log_dir Coop/outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -use_attr -n_attributes
$N_ATTR -attr_loss_weight 0.010 -normalize_loss -b 64 -weight_decay 0.00004 -lr 0.01 -scheduler_step 20 -dset $DATASET -data_dir $DATA_DIR
 ```



