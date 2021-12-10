# Simple Time Series Prediction in Pytorch Using LSTM and GRU

## Table of contents

- [Models](#models)
- [Quick Start](#quick-start)
- [Creators](#creators)
- [Copyright and license](#copyright-and-license)

## Models

Long Short Term Memory:

```
LSTM(
  (lstm): LSTM(1, 100)
  (fc1): Linear(in_features=100, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=1, bias=True)
  (relu): ReLU()
)
```

Gated Recurrent Unit:

```
GRU(
  (lstm): GRU(1, 100)
  (fc1): Linear(in_features=100, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=1, bias=True)
  (relu): ReLU()
)
```

## Quick Start

Install torch and torchvision using pip:

```
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

Install the dependencies:

```
pip install -r requirements.txt
```

Run one the examples:

```
python sin_prediction_gpu.py
```


## Creators

**Federico Zappone**

- <https://github.com/federicozappone>

## Copyright and license

Code released under the [MIT License](https://github.com/federicozappone/rover/LICENSE.md).
