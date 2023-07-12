### How to run

```commandline
python index.py fit --trainer.default_root_dir 'logs/' --trainer.max_epochs 6 --data.csv_file './dataset/train.csv' --model.classes 10

python index.py test --trainer.default_root_dir 'logs/' --trainer.max_epochs 6 --data.csv_file './dataset/train.csv' --model.classes 10
```