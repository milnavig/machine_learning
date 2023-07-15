### How to run

```commandline
python index.py fit --trainer.default_root_dir 'logs/' --trainer.max_epochs 6 --data.csv_file './dataset/train.csv' --model.classes 10 --model.number_internal_neurons 64

python index.py test --trainer.default_root_dir 'logs/' --ckpt_path 'logs/lightning_logs/version_2/checkpoints/epoch=14-step=7875.ckpt' --trainer.max_epochs 6 --data.csv_file './dataset/train.csv' --model.classes 10 --model.number_internal_neurons 64 
```

### Launch Tensorboard

```commandline
tensorboard --logdir=logs
```