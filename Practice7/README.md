### How to run

```commandline
python index.py fit --model GRU --trainer.default_root_dir 'logs/' --trainer.max_epochs 100 --data.csv_file './dataset/output.csv' --model.classes 5 --model.input_size 200 --model.hidden_size 300

```

### Launch Tensorboard

```commandline
tensorboard --logdir=logs
```