import sys
sys.path.append('/data/sjb_workspace/unsupervise_rl/url_benchmark-main/lexa_env/Metaworld')
from metaworld.envs.mujoco.env_dict import (
    ALL_V2_ENVIRONMENTS_GOAL_HIDDEN,
    ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
)

__all__ = ["ALL_V2_ENVIRONMENTS_GOAL_HIDDEN", "ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE"]
