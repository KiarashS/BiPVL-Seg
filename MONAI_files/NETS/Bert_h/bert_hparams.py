from dataclasses import dataclass
from typing import List

import json
from dataclasses import dataclass

@dataclass
class BERTHyperParams:
    @classmethod
    def from_json(cls, fpath):
        with open(fpath, "r") as f:
            data = json.load(f)
        return cls(**data)

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    # Method
    layers: List[int]
    fact_token: str
    v_num_grad_steps: int
    v_lr: float
    v_loss_layer: int
    v_weight_decay: float
    clamp_norm_factor: float
    kl_factor: float

    # Module templates
    rewrite_module_tmp: str
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    ln_f_module: str
    lm_head_module: str

    # Projection parameters
    nullspace_dimension: int
    update: str
    projection_location: str

    batch_size: int
    orthogonal_constraint: float
