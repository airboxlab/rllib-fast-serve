import argparse
import ast
import json
import os
from typing import Dict, Any, Optional, Union, List

import numpy
import numpy as np
import onnxruntime as rt

parser = argparse.ArgumentParser()
parser.add_argument(
    "--onnx-model",
    help="Path to .onnx model file",
    required=True
)
parser.add_argument(
    "--graph-io",
    help="Path to graph_io.json. This will be used to restore inputs/outputs structure of original model",
    required=True
)
parser.add_argument(
    "--filters",
    help="Path to filters.json. This will be used to pre-process observations",
    required=False
)
parser.add_argument(
    "--obs",
    help="The observations vector (ie --obs \"[0, 1, 2]\")",
    type=ast.literal_eval, required=True
)


def print_onnx_model_graph(model_path: str):
    """for debugging"""
    import onnx
    from google.protobuf.json_format import MessageToDict
    model = onnx.load(model_path)
    for _input in model.graph.input:
        print(MessageToDict(_input))
    for _output in model.graph.output:
        print(MessageToDict(_output))


def load_filters(filters_path: Optional[str]):
    """load observations filters"""
    filters = {}
    if filters_path:
        with open(filters_path, "r") as f:
            filters = json.load(f)
    return filters


def load_graph_io(graphio_path: str):
    """load input/output structure of graph"""
    with open(graphio_path, "r") as f:
        graph_io = json.load(f)
    return graph_io


def infer(config: Dict[str, str], observations: Dict[str, Union[List[float], np.ndarray]]) -> Dict[str, Any]:
    model_path = config["model_path"]
    assert os.path.exists(model_path)
    graphio = load_graph_io(config["graphio_path"])
    filters = load_filters(config.get("filters_path"))
    actions = {}

    for policy_id, obs in observations.items():

        if filters:
            f = filters[policy_id]
            if f:
                if f["demean"]:
                    obs = obs - np.array(f["mean"], dtype=numpy.float32)
                if f["destd"]:
                    obs = obs / np.array(f["std"], dtype=numpy.float32)
                if f["clip"]:
                    obs = np.clip(obs, -f["clip"], f["clip"])

        inputs = {f"{policy_id}/obs:0": numpy.array([obs], dtype=numpy.float32)}
        outputs = {k: v["name"] for k, v in graphio["outputs"].items()}

        sess = rt.InferenceSession(model_path)
        actions[policy_id] = sess.run(output_names=[outputs["actions_0"]], input_feed=inputs)

    return actions


if __name__ == "__main__":
    args = parser.parse_args()

    cfg = {
        "model_path": args.onnx_model,
        "graphio_path": args.graph_io,
        "filters_path": args.filters,
    }
    obs_input = args.obs
    assert isinstance(obs_input, list)

    predicted = infer(config=cfg, observations={"default_policy": obs_input})
    print(predicted)
