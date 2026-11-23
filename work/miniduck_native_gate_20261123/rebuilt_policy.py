"""CPU inference adapter for the actor reconstructed from ONNX weights."""
import json
from pathlib import Path

import numpy as np

from portable_policy import policy_type

Base = policy_type()


class RebuiltHeadingPolicy(Base):
    @classmethod
    def load(cls, params_path, contract_path):
        params_path, contract_path = Path(params_path), Path(contract_path)
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        obj = cls(contract)
        z = np.load(params_path)
        obj.params = {k: np.asarray(z[k], dtype=np.float32) for k in z.files}
        if obj.params["mean"].shape != (75,) or obj.params["std"].shape != (75,):
            raise ValueError("Invalid reconstructed normalizer")
        return obj

    def infer(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        if obs.shape != (75,) or not np.isfinite(obs).all():
            raise ValueError("Invalid observation")
        x = (obs - self.params["mean"]) / self.params["std"]
        for w, b in (("w0", "b0"), ("w1", "b1"), ("w2", "b2")):
            z = x @ self.params[w] + self.params[b]
            x = z / (1.0 + np.exp(-z))
        z = x @ self.params["w3"] + self.params["b3"]
        action = np.tanh(z[:10]).astype(np.float32)
        if np.max(np.abs(action)) > 1.00001:
            raise ValueError("Rebuilt actor violated bounded output contract")
        return np.clip(action, -1., 1.)
