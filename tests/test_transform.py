import json
import shutil
import unittest
import tempfile
from typing import Dict, Any, Optional, Type

import ray.rllib.agents.ppo as ppo

from src.infer import infer, apply_filter
from src.rllib2onnx import (
    restore,
    SimplePolicyConfigAdapter,
    rllib_to_onnx,
    PolicyConfigAdapter
)


class TestTransform(unittest.TestCase):

    def test_train_transform_infer(self):
        self._test(
            base_output=tempfile.mkdtemp(),
            env="CartPole-v0"
        )

    def test_filter_0_std(self):
        filtered = apply_filter(
            filt={
                "demean": True, "destd": True, "clip": None,
                "mean": 1, "std": 0
            },
            obs=[1]
        )
        self.assertEqual([0], filtered)

    def test_filter(self):
        base_output = tempfile.mkdtemp()
        self._test(
            base_output=base_output,
            env="CartPole-v0",
            trainer_cfg={"observation_filter": "MeanStdFilter"}
        )

        with open(f"{base_output}/filters/filters.json", "r") as f_file:
            f_def = json.load(f_file)
            f_pol = f_def["default_policy"]
            self.assertIsNotNone(f_pol)
            for key in ["shape", "demean", "destd", "mean", "std"]:
                self.assertIn(key, f_pol)

    def test_config_adapter(self):

        class FailingPolicyConfigAdapter(SimplePolicyConfigAdapter):
            def adapt(
                    self,
                    config_override: Optional[Dict[str, Any]] = None
            ) -> Dict[str, Any]:
                cfg = super().adapt(config_override)
                return {"unknown": "bla", **cfg}

        with self.assertRaises(Exception) as e:
            self._test(
                base_output=tempfile.mkdtemp(),
                env="CartPole-v0",
                config_adapter_cls=FailingPolicyConfigAdapter
            )
            self.assertTrue("unknown config parameter" in str(e.exception).lower())

    def _test(
            self,
            base_output: str,
            env: Optional[str] = None,
            trainer_cfg: Optional[Dict[str, Any]] = None,
            config_adapter_cls: Optional[Type[PolicyConfigAdapter]] = None
    ):
        # Train on CartPole for one iteration
        trainer = ppo.PPOTrainer(
            config={
                "framework": "tf",
                "num_workers": 0,
                **(trainer_cfg or {})
            },
            env=env,
        )
        trainer.train()
        trainer.save("/tmp/rllib_checkpoint")
        params = f"{trainer.logdir}/params.pkl"
        chkpt_dir = "/tmp/rllib_checkpoint/checkpoint_000001/"
        shutil.copy(params, chkpt_dir)
        checkpoint_path = f"{chkpt_dir}/checkpoint-1"

        # restore the trained policy
        restored_trainer = restore(
            alg="PPO",
            checkpoint_file=checkpoint_path,
            env=env,
            config_adapter_cls=config_adapter_cls
        )

        # output an ONNX model
        rllib_to_onnx(
            agent=restored_trainer,
            output_dir=base_output
        )

        # run inference
        infer_cfg = {
            "model_path": f"{base_output}/onnx/saved_model.onnx",
            "graphio_path": f"{base_output}/graph_io/graph_io.json",
            "filters_path": f"{base_output}/filters/filters.json",
        }
        obs = trainer.get_policy().observation_space.sample()
        act = infer(config=infer_cfg, observations={"default_policy": obs})
        print(f"output: {act}")
        self.assertTrue(trainer.get_policy().action_space.contains(act["default_policy"][0][0]))
