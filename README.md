# 4M25

Information on how to run things will go here.

## How to run training

1) Go to `trainer.py` and adjust training parameters in the `get_parameters()` function.

2) Run
```python
python main.py
```

3) Model checkpoint and GIF will be saved to the corresponding folder in the `logs` directory.

4) The training metrics can be monitored _during_ or after training using
```
tensorboard --logdir <save_directory>
```
e.g.
```
tensorboard --logdir logs/baseline_2023_02_26_14_13_59
```

Tensorboard can be installed using
```
pip install tensorboard=2.12.0
```
