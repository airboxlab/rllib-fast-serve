"""
transforms a checkpoint into a tensorflow model and an ONNX model
"""
import argparse
import ast
import copy
import json
import logging
import os
import pickle
from abc import abstractmethod, ABCMeta
from typing import Dict, Any, Type, Optional

import gym
import ray
import tensorflow
from ray.rllib import ExternalEnv
from ray.rllib.agents import Trainer
from ray.rllib.utils import Filter
from ray.rllib.utils.filter import NoFilter, MeanStdFilter
from ray.tune.registry import get_trainable_cls, register_env

parser = argparse.ArgumentParser()
parser.add_argument(
    "--alg", help="Algorithm used during training (ie PPO, DQN, SAC, ...)", required=True
)
parser.add_argument(
    "--checkpoint-file", help="Checkpoint to restore", required=True
)
parser.add_argument(
    "--config",
    help="Override reloaded config with --config '{\"[key]\":\"[value]\"}'",
    type=ast.literal_eval, default={}, required=False
)
parser.add_argument(
    "--output-dir", help="output directory", default="/tmp/model_export"
)

ray.init(
    include_dashboard=False,
    ignore_reinit_error=True
)


class PolicyConfigAdapter(metaclass=ABCMeta):
    """a helper that adapts a reloaded training config for inference"""

    def __init__(self, loaded_config: Dict[str, Any]):
        self._adapted_config = copy.deepcopy(loaded_config)

    def adapt(
            self,
            config_override: Optional[Dict[str, Any]] = None):
        self._adapted_config["num_workers"] = 1
        self._adapted_config["num_gpus"] = 0
        self._adapted_config.update(config_override or {})
        return self._adapted_config

    @abstractmethod
    def obs_space(self) -> gym.spaces.Space:
        pass

    @abstractmethod
    def act_space(self) -> gym.spaces.Space:
        pass


class SimplePolicyConfigAdapter(PolicyConfigAdapter):
    def obs_space(self) -> gym.spaces.Space:
        pass

    def act_space(self) -> gym.spaces.Space:
        pass


def register_ext_env(env_name: str, obs_space: gym.spaces.Space, act_space: gym.spaces.Space):
    """create and register a dummy external env"""

    class PseudoEnv(ExternalEnv):
        def __init__(self, observation_space, action_space):
            ExternalEnv.__init__(
                self,
                action_space=action_space,
                observation_space=observation_space
            )

        def run(self):
            pass

    register_env(
        env_name,
        lambda _: PseudoEnv(
            observation_space=obs_space,
            action_space=act_space
        )
    )


def restore_config_pkl(checkpoint_file: str) -> Dict[str, Any]:
    """
    reload params.pkl that is saved in experiment dir.
    assuming here that .pkl is in same dir than checkpoint
    """
    config_dir = os.path.dirname(checkpoint_file)
    config_path = os.path.join(config_dir, "params.pkl")
    if not os.path.exists(config_path):
        logging.warning(f"couldn't find {config_path}")
        return {}
    else:
        with open(config_path, "rb") as f:
            config = pickle.load(f)
        return config


def restore(
        alg: str,
        checkpoint_file: str,
        config_adapter_cls: Type[PolicyConfigAdapter],
        cfg_override: Optional[Dict[str, Any]] = None,
        env: Optional[str] = None
) -> Trainer:
    """reload agent from config class and checkpoint"""
    config = restore_config_pkl(checkpoint_file)

    config_adapter = config_adapter_cls(config)
    config = config_adapter.adapt(config_override=cfg_override)

    env_name = env
    if env_name is None:
        env_name = "transient_env"
        obs_space = config_adapter.obs_space()
        act_space = config_adapter.act_space()
        assert isinstance(obs_space, gym.spaces.Space)
        assert isinstance(act_space, gym.spaces.Space)
        register_ext_env(env_name, obs_space, act_space)

    # note: if you have custom policy model, action distributions
    # or policy classes, they should be registered here
    # see https://docs.ray.io/en/latest/rllib/rllib-models.html
    cls = get_trainable_cls(alg)
    agent: Trainer = cls(env=env_name, config=config)
    agent.restore(checkpoint_file)

    return agent


def dump_filters(agent: Trainer, base_output: str) -> None:
    """
    dump observation filters (for each policy).
    filters are considered frozen and won't be updated with inference
    """
    filters: Dict[str, Filter] = agent.workers.local_worker().filters
    ser_filters = {}

    for policy_id, filt in filters.items():
        if isinstance(filt, NoFilter):
            ser_filters[policy_id] = None
        elif isinstance(filt, MeanStdFilter):
            ser_filters[policy_id] = {
                "shape": filt.shape.tolist(),
                "demean": filt.demean,
                "destd": filt.destd,
                "clip": filt.clip,
                "mean": filt.rs.mean.tolist(),
                "std": filt.rs.std.tolist()
            }

    filter_dir = os.path.join(base_output, "filters")
    os.makedirs(filter_dir, exist_ok=False)
    filters_file = os.path.join(filter_dir, "filters.json")
    with open(filters_file, "w") as f:
        json.dump(ser_filters, f)


def dump_graph_structure(tf_dir: str, base_output: str) -> None:
    """export input and output structure of graph"""
    loaded = tensorflow.saved_model.load(tf_dir)
    sig = loaded.signatures["serving_default"]

    # note: for input tensors, tensor name == spec name
    graph = {
        "inputs": {
            k: {
                "name": v.name,
                "shape": v.shape.as_list()
            }
            for k, v
            in sig.structured_input_signature[1].items()
        },
        "outputs": {
            k: {
                "name": v.name,
                "shape": v.shape.as_list()
            }
            for k, v
            in sig.structured_outputs.items()
        }
    }

    io_dir = os.path.join(base_output, "graph_io")
    os.makedirs(io_dir, exist_ok=False)
    io_file = os.path.join(io_dir, "graph_io.json")
    with open(io_file, "w") as f:
        json.dump(graph, f)


def rllib_to_onnx(
        agent: Trainer,
        output_dir: str
) -> None:
    """export tasks"""
    base_output = output_dir

    # export TF model
    tf_dir = os.path.join(base_output, "tf")
    agent.export_policy_model(tf_dir)

    # extract graph structure
    dump_graph_structure(tf_dir, base_output)

    # export ONNX
    onnx_dir = os.path.join(base_output, "onnx")
    agent.export_policy_model(onnx_dir, onnx=1)

    # export filters
    dump_filters(agent, base_output)


if __name__ == "__main__":
    args = parser.parse_args()
    assert os.path.exists(args.checkpoint_file)


    class PolicyConfigWithSpaces(PolicyConfigAdapter):
        """
        simple config adapter that extracts observation and action
        spaces from config reloaded from .pkl
        """

        def obs_space(self):
            return self._adapted_config["env_config"]["observation_space"]

        def act_space(self):
            return self._adapted_config["env_config"]["action_space"]


    a = restore(
        alg=args.alg,
        checkpoint_file=args.checkpoint_file,
        config_adapter_cls=PolicyConfigWithSpaces,
        cfg_override=args.config
    )

    rllib_to_onnx(
        agent=a,
        output_dir=args.output_dir
    )
