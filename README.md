# rllib-fast-serve

Tools and examples to export policies trained with [Ray RLlib](https://github.com/ray-project/ray) for lightweight and
fast inference.

Only `tensorflow` supported for now, but adding support for `pytorch` should be fairly easy.

# Usage

```shell
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Notes on dependencies:

- `ray`, `tensorflow` and `tf2onnx` aren't desired for inference (only needed for phase 1).
  If you build an inference tool/server, make sure to pull them out!
- `ray` and `tensorflow` versions depend on the ones you used for training.
- `numpy` version must align with requirements of `tensorflow` and `onnx`. If you need to change it, check their
  respective compatibility matrix

## Phase 1: transform

This will transform an input RLlib checkpoint to several artifacts needed for inference.

```shell
python3 src/rllib2onnx.py --alg PPO --checkpoint-file /tmp/my_training/checkpoint-1
```

`rllib2onnx` can also reload training config from `.pkl`, assuming that it's located in checkpoint directory. See code
for usage.

Output listing:

- a TF saved model (`.pb`).
- an ONNX model (`.onnx`)
- a json containing input and output nodes structure of the model graph.
- a json containing observations filters definition (needed to standardize observations if you used `MeanStdFilter`
  during training).

Example output directory structure:

```shell
/tmp/model_export/
├── filters
│   └── filters.json
├── graph_io
│   └── graph_io.json
├── onnx
│   └── saved_model.onnx
└── tf
    ├── events.out.tfevents.1657097632.work
    ├── saved_model.pb
    └── variables
        ├── variables.data-00000-of-00001
        └── variables.index
```

## Phase 2: serve

Run model inference using ONNX model

```shell
python3 src/infer.py \
  --onnx-model /tmp/model_export/onnx/saved_model.onnx \
  --filters /tmp/model_export/filters/filters.json \
  --graph-io /tmp/model_export/graph_io/graph_io.json \
  --obs "[1, 2, 3]"
```

Example output:

```
{'default_policy': [array([[14, 33]], dtype=int64)]}
```
