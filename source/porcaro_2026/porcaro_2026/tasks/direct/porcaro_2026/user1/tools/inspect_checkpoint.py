#!/usr/bin/env python3
"""Inspect an RSL-RL checkpoint and its saved run configuration without Isaac Sim."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

try:
    import torch
except Exception as exc:  # pragma: no cover - environment dependent
    raise SystemExit(f"Could not import torch: {exc}")

try:
    import yaml
except Exception as exc:  # pragma: no cover - environment dependent
    raise SystemExit(f"Could not import PyYAML: {exc}")


class RunConfigLoader(yaml.SafeLoader):
    """Safe YAML loader that tolerates Hydra's Python tuple/slice tags."""


def _python_tag(loader: RunConfigLoader, suffix: str, node: yaml.Node) -> Any:
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    return loader.construct_scalar(node)


RunConfigLoader.add_multi_constructor("tag:yaml.org,2002:python/", _python_tag)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def first_weight_shape(obj: Any, suffixes: tuple[str, ...]) -> tuple[str, list[int]] | None:
    """Find a 2-D tensor weight by key suffix, searching nested state dictionaries."""
    seen: set[int] = set()

    def walk(value: Any, prefix: str = "") -> tuple[str, list[int]] | None:
        identity = id(value)
        if identity in seen:
            return None
        if isinstance(value, dict):
            seen.add(identity)
            for key, child in value.items():
                key_text = str(key)
                full = f"{prefix}.{key_text}" if prefix else key_text
                if key_text.endswith(suffixes) and hasattr(child, "shape") and len(child.shape) == 2:
                    return full, list(child.shape)
            for key, child in value.items():
                result = walk(child, f"{prefix}.{key}" if prefix else str(key))
                if result:
                    return result
        return None

    return walk(obj)


def get_nested(data: Any, *keys: str) -> Any:
    value = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Path to a RSL-RL .pt checkpoint")
    parser.add_argument("--expect-lookahead", type=float)
    parser.add_argument("--expect-obs", type=int)
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        print(f"ERROR: checkpoint not found: {checkpoint}", file=sys.stderr)
        return 2

    cfg_path = checkpoint.parent / "params" / "env.yaml"
    cfg: dict[str, Any] | None = None
    cfg_error = None
    if cfg_path.is_file():
        try:
            loaded = yaml.load(cfg_path.read_text(encoding="utf-8"), Loader=RunConfigLoader)
            cfg = loaded if isinstance(loaded, dict) else None
        except Exception as exc:
            cfg_error = f"{type(exc).__name__}: {exc}"
    else:
        cfg_error = "same-run params/env.yaml not found"

    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except Exception as exc:
        print(f"ERROR: weights_only checkpoint read failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    actor = first_weight_shape(state, ("actor.0.weight", "actor.0.weight_ih_l0"))
    critic = first_weight_shape(state, ("critic.0.weight", "critic.0.weight_ih_l0"))
    # Recurrent RSL-RL commonly stores first affine layers under actor/critic.0.weight.
    # Prefer explicit recurrent input weights if present.
    actor_rnn = first_weight_shape(state, ("actor_rnn.weight_ih_l0", "memory_a.rnn.weight_ih_l0", "actor_memory.rnn.weight_ih_l0"))
    critic_rnn = first_weight_shape(state, ("critic_rnn.weight_ih_l0", "memory_c.rnn.weight_ih_l0", "critic_memory.rnn.weight_ih_l0"))

    iteration = None
    if isinstance(state, dict):
        iteration = state.get("iter", state.get("iteration"))
        if hasattr(iteration, "item"):
            iteration = iteration.item()

    env_obs = cfg.get("observation_space") if cfg else None
    env_lookahead = cfg.get("lookahead_horizon") if cfg else None
    sim_dt = get_nested(cfg, "sim", "dt") if cfg else None
    decimation = cfg.get("decimation") if cfg else None
    controller = cfg.get("controller") if cfg and isinstance(cfg.get("controller"), dict) else None
    actor_shape = (actor_rnn or actor)
    critic_shape = (critic_rnn or critic)
    actor_in = actor_shape[1][1] if actor_shape and len(actor_shape[1]) > 1 else None
    critic_in = critic_shape[1][1] if critic_shape and len(critic_shape[1]) > 1 else None
    weight_dims_agree = actor_in is not None and critic_in is not None and actor_in == critic_in
    cfg_dims_agree = env_obs is not None and actor_in == env_obs and critic_in == env_obs

    report = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "iteration": iteration,
        "run_env_yaml": str(cfg_path) if cfg_path.is_file() else None,
        "run_config_error": cfg_error,
        "run_config": {
            "lookahead_horizon": env_lookahead,
            "observation_space": env_obs,
            "sim_dt": sim_dt,
            "decimation": decimation,
            "controller": controller,
        } if cfg is not None else None,
        "actor_weight_ih_l0_or_first_weight": {"key": actor[0], "shape": actor[1]} if actor else None,
        "critic_weight_ih_l0_or_first_weight": {"key": critic[0], "shape": critic[1]} if critic else None,
        "actor_recurrent_weight_ih_l0": {"key": actor_rnn[0], "shape": actor_rnn[1]} if actor_rnn else None,
        "critic_recurrent_weight_ih_l0": {"key": critic_rnn[0], "shape": critic_rnn[1]} if critic_rnn else None,
        "weight_input_dimensions_agree": weight_dims_agree,
        "run_config_matches_weight_input_dimension": cfg_dims_agree,
        "expectations": {
            "lookahead": args.expect_lookahead,
            "observation_space": args.expect_obs,
            "lookahead_matches": args.expect_lookahead is None or env_lookahead == args.expect_lookahead,
            "observation_matches": args.expect_obs is None or env_obs == args.expect_obs,
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))

    if cfg is None or actor_in is None or critic_in is None or not weight_dims_agree or not cfg_dims_agree:
        return 1
    if args.expect_lookahead is not None and env_lookahead != args.expect_lookahead:
        return 1
    if args.expect_obs is not None and env_obs != args.expect_obs:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
