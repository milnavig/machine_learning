### How to run

```commandline
python index.py fit --trainer.default_root_dir 'logs/' --trainer.max_epochs 6 --data.csv_file './dataset/output.csv' --model.classes 3 --model.input_size 200 --model.hidden_size 600

python index.py test --trainer.default_root_dir 'logs/' --ckpt_path 'logs/lightning_logs/version_0/checkpoints/epoch=5-step=42.ckpt' --trainer.max_epochs 6 --data.csv_file './dataset/output.csv' --model.classes 3 --model.input_size 200 --model.hidden_size 600
```

### Launch Tensorboard

```commandline
tensorboard --logdir=logs
```