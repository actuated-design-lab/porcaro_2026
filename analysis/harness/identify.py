"""Condition classification from recorded run params (env.yaml + agent.yaml).

No isaaclab / isaacsim / omni / torch imports. Pure-Python + PyYAML only.

NOTE on labels: classify_model() returns short internal labels ("A".."E") for
the five conditions actually swept by scripts/rsl_rl/run_experiment_matrix.py
(LSTM lookahead 0.1/0.5/1.0, plain MLP @0.5, frame-stacked MLP @0.5). These
labels are the experiment matrix's own bookkeeping labels and are NOT
guaranteed to line up with the paper's "Model A/B/C/D" lettering - do not
conflate the two without checking scripts/rsl_rl/run_experiment_matrix.py's
build_conditions() against the paper text.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class _PorcaroSafeLoader(yaml.SafeLoader):
    """yaml.SafeLoader plus narrow constructors for a couple of Python tags.

    isaaclab's dump_yaml() (used by scripts/rsl_rl/train.py to write
    params/env.yaml) serializes plain Python tuples and `slice` objects
    (e.g. from ContactSensorCfg/ArticulationCfg fields) using PyYAML's
    full-dump tags (`tag:yaml.org,2002:python/tuple`,
    `tag:yaml.org,2002:python/object/apply:builtins.slice`), which plain
    yaml.safe_load() refuses to resolve by design. Rather than switching to
    yaml.unsafe_load()/yaml.full_load() (which would construct *any* tagged
    Python object, including arbitrary callables), we register constructors
    for only this small, explicit allowlist of known-benign builtin types.
    Any other `!!python/...` tag still raises PyYAML's normal
    ConstructorError instead of silently executing something unexpected.
    """


def _construct_python_tuple(loader: yaml.SafeLoader, node: yaml.Node):
    return tuple(loader.construct_sequence(node))


def _construct_builtins_slice(loader: yaml.SafeLoader, node: yaml.Node):
    args = loader.construct_sequence(node)
    return slice(*args)


_PorcaroSafeLoader.add_constructor("tag:yaml.org,2002:python/tuple", _construct_python_tuple)
_PorcaroSafeLoader.add_constructor(
    "tag:yaml.org,2002:python/object/apply:builtins.slice", _construct_builtins_slice
)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a small params YAML (env.yaml / agent.yaml) as a plain dict."""
    with open(path, "r") as f:
        return yaml.load(f, Loader=_PorcaroSafeLoader)


def classify_model(env_y: dict, agent_y: dict) -> str | None:
    """Map a run's (lookahead_horizon, frame_stacking, policy type) to a label.

    Reproduced verbatim from the task spec - do not change the mapping logic.
    """
    lh = round(float(env_y["lookahead_horizon"]), 3)
    fs = bool(env_y.get("use_frame_stacking", False))
    pol = agent_y["policy"]
    recurrent = pol.get("class_name") == "ActorCriticRecurrent" or bool(pol.get("rnn_type"))
    if recurrent and not fs:
        return {0.1: "A", 0.5: "B", 1.0: "C"}.get(lh)
    if not recurrent and not fs and lh == 0.5:
        return "D"
    if not recurrent and fs and lh == 0.5:
        return "E"
    return None  # 想定外の組み合わせ


# Cross-reference table: run_experiment_matrix.py's exp_tag -> classify_model() label.
# Built from build_conditions() in scripts/rsl_rl/run_experiment_matrix.py:
#   lstm_lookahead0.1            -> recurrent, lh=0.1           -> "A"
#   lstm_lookahead0.5            -> recurrent, lh=0.5           -> "B"
#   lstm_lookahead1.0            -> recurrent, lh=1.0           -> "C"
#   mlp_plain_lookahead0.5       -> not recurrent, no fs, lh=0.5 -> "D"
#   mlp_framestack_k5_lookahead0.5 -> not recurrent, fs, lh=0.5  -> "E"
EXP_TAG_TO_MODEL: dict[str, str] = {
    "lstm_lookahead0.1": "A",
    "lstm_lookahead0.5": "B",
    "lstm_lookahead1.0": "C",
    "mlp_plain_lookahead0.5": "D",
    "mlp_framestack_k5_lookahead0.5": "E",
}
