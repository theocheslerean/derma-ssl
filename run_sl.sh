#!/bin/sh

python train.py fit --config config_sl.yaml

# python train.py fit --config config_sl.yaml --seed_everything 15 --trainer.max_epochs=100
# python train.py fit --config config_sl.yaml --seed_everything 42 --trainer.max_epochs=100
# python train.py fit --config config_sl.yaml --seed_everything 100 --trainer.max_epochs=100
# python train.py fit --config config_sl.yaml --seed_everything 1024 --trainer.max_epochs=100