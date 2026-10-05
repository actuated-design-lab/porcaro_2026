"""Fail-fast checks and provenance for the user1 SI2026 training run."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import math
import shutil
import subprocess
import sys

import torch

EXPECTED_PACKAGE = Path('/home/labuser1/IsaacLab/source/porcaro_2026/porcaro_2026').resolve()
EXPECTED_USER = EXPECTED_PACKAGE / 'tasks/direct/porcaro_2026/user1'


def require(condition, message):
    if not condition:
        raise RuntimeError('[SI2026 verification FAILED] ' + message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def verify_config(cfg, agent_cfg, args):
    import porcaro_2026
    require(Path(porcaro_2026.__file__).resolve().parent == EXPECTED_PACKAGE,
            f'Wrong package source: {porcaro_2026.__file__}')
    require(Path(inspect.getfile(type(cfg))).resolve().parent == EXPECTED_USER,
            'Environment configuration did not come from labuser1/user1')
    require(args.task == 'Template-Porcaro-2026-ModelB-DR-user1', 'Wrong Gym task')
    require(math.isclose(cfg.lookahead_horizon, 0.5), 'lookahead_horizon must be 0.5 s')
    require(cfg.observation_space == 35, 'observation_space must be 35')
    require(math.isclose(cfg.sim.dt * cfg.decimation, 0.02), 'Control period must be 0.02 s')
    require(cfg.action_space == 3, 'Expected three PAM channels')
    require(cfg.controller.control_mode == 'pressure', 'Expected pressure action mode')
    require(math.isclose(cfg.controller.Pmax, 0.6), 'Pmax must be 0.6 MPa')
    require(math.isclose(cfg.controller.discrete_threshold, 0.5), 'Expected 50% switching threshold')
    require(cfg.controller.use_discrete_action, 'Binary action flag must be enabled')
    require(not agent_cfg.resume, 'This verified entry point starts a new training run')
    require(agent_cfg.policy.rnn_type == 'lstm', 'Expected LSTM policy')
    print('[SI2026] CONFIG PASS: own user1 source, lookahead=0.5 s, obs=35, pressure=0/0.6 MPa', flush=True)


def verify_runtime(env, runner, cfg, agent_cfg, log_dir, argv):
    plant = env.unwrapped
    require(Path(inspect.getfile(type(plant))).resolve().parent == EXPECTED_USER,
            'Runtime environment class came from the wrong source')
    require(type(plant.action_controller).__name__ == 'DiscreteTorqueActionController',
            'Runtime controller is not binary')
    require(plant.lookahead_steps == 25, 'Runtime lookahead must contain 25 samples')
    obs = plant._get_observations()['policy']
    require(tuple(obs.shape) == (plant.num_envs, 35), f'Actual observation shape: {tuple(obs.shape)}')
    require(bool(torch.isfinite(obs).all()), 'Nonfinite initial observation')
    state = runner.alg.policy.state_dict()
    shapes = {key: list(state[key].shape) for key in
              ('memory_a.rnn.weight_ih_l0', 'memory_c.rnn.weight_ih_l0')}
    require(all(shape[1] == 35 for shape in shapes.values()), f'LSTM input dimensions: {shapes}')

    # Pure action conversion: it does not step physics, consume RNG or alter PAM state.
    probe = torch.tensor([[-1., -1., -1.], [-0.01, 0., 0.01], [1., 1., 1.]], device=plant.device)
    pressure = plant.action_controller.compute_pressure(probe)
    expected = torch.tensor([[0., 0., 0.], [0., 0.6, 0.6], [0.6, 0.6, 0.6]], device=plant.device)
    require(bool(torch.allclose(pressure, expected, atol=1e-6, rtol=0)), 'Pressure conversion probe failed')

    log_dir = Path(log_dir).resolve()
    snapshots = log_dir / 'source_snapshot'
    hashes = {}
    for folder in (EXPECTED_USER, EXPECTED_USER.parent / 'common'):
        for source in sorted(folder.rglob('*.py')):
            if '__pycache__' in source.parts:
                continue
            relative = source.relative_to(EXPECTED_PACKAGE)
            target = snapshots / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            hashes[str(relative)] = sha256(source)
    assets = {}
    for source in (cfg.robot_cfg.spawn.usd_path, cfg.drum_cfg.spawn.usd_path,
                   cfg.controller.force_map_csv, cfg.controller.h0_map_csv):
        path = Path(source).resolve()
        require(path.is_relative_to(EXPECTED_PACKAGE), f'Asset outside own package: {path}')
        assets[str(path)] = sha256(path)
    report = {
        'verified_at_utc': datetime.now(timezone.utc).isoformat(),
        'run_dir': str(log_dir), 'python': sys.executable, 'command_argv': argv,
        'environment_source': inspect.getfile(type(plant)),
        'configuration_source': inspect.getfile(type(cfg)),
        'controller_source': inspect.getfile(type(plant.action_controller)),
        'lookahead_horizon_s': cfg.lookahead_horizon, 'lookahead_samples': plant.lookahead_steps,
        'control_dt_s': cfg.sim.dt * cfg.decimation, 'actual_observation_shape': list(obs.shape),
        'lstm_input_weights': shapes, 'pressure_low_mpa': 0.0, 'pressure_high_mpa': 0.6,
        'pressure_threshold_mpa': cfg.controller.Pmax * cfg.controller.discrete_threshold,
        'pressure_probe_actions': probe.cpu().tolist(), 'pressure_probe_outputs': pressure.cpu().tolist(),
        'num_envs': plant.num_envs, 'seed': cfg.seed, 'max_iterations': agent_cfg.max_iterations,
        'resume': agent_cfg.resume, 'source_sha256': hashes, 'asset_sha256': assets,
        'versions': {name: importlib.metadata.version(name) for name in ['torch', 'rsl-rl-lib', 'porcaro_2026']},
    }
    write_json(log_dir / 'verified_runtime.json', report)
    print(f'[SI2026] RUNTIME PASS: obs={tuple(obs.shape)}, lookahead=25, LSTM={shapes}', flush=True)
    print(f'[SI2026] VERIFIED RUN: {log_dir}', flush=True)

    # Check telemetry from the actual physics path, not just the standalone converter.
    original_apply = plant._apply_action
    samples = []
    def checked_apply():
        original_apply()
        telemetry = plant.action_controller.get_last_telemetry()
        command, output = telemetry['P_cmd'], telemetry['P_out']
        valid = torch.isclose(command, torch.zeros_like(command), atol=1e-6, rtol=0) | torch.isclose(
            command, torch.full_like(command, 0.6), atol=1e-6, rtol=0)
        require(bool(valid.all()), 'Nonbinary command in real physics telemetry')
        require(bool(torch.isfinite(output).all()), 'Nonfinite PAM pressure response')
        require(bool(((output >= -1e-6) & (output <= 0.600001)).all()), 'PAM pressure out of range')
        samples.append({'physics_step': len(samples) + 1, 'P_cmd_env0': command[0].cpu().tolist(),
                        'P_out_env0': output[0].cpu().tolist()})
        if len(samples) == 64:
            write_json(log_dir / 'verified_pressure_samples.json', {
                'checked_physics_steps': 64, 'checked_envs': plant.num_envs,
                'P_cmd_allowed_mpa': [0.0, 0.6], 'samples': samples})
            print('[SI2026] PHYSICS PASS: binary P_cmd checked for 64 physics steps in all environments', flush=True)
            plant._apply_action = original_apply
    plant._apply_action = checked_apply

    # Each saved checkpoint carries compact provenance and is checked after writing.
    original_save = runner.save
    def checked_save(path, infos=None):
        info = dict(infos or {})
        info['si2026_verified'] = {
            'lookahead_horizon_s': 0.5, 'observation_dim': 35,
            'pressure_command_mpa': [0.0, 0.6], 'run_dir': str(log_dir),
            'runtime_manifest_sha256': sha256(log_dir / 'verified_runtime.json'),
        }
        original_save(path, infos=info)
        saved = torch.load(path, map_location='cpu', weights_only=True)
        dims = {key: list(saved['model_state_dict'][key].shape) for key in shapes}
        require(all(shape[1] == 35 for shape in dims.values()), 'Saved checkpoint is not 35-dimensional')
        write_json(Path(path).with_suffix('.verification.json'), {
            **info['si2026_verified'], 'iteration': saved.get('iter'),
            'checkpoint': str(Path(path).resolve()), 'checkpoint_sha256': sha256(path),
            'lstm_input_weights': dims})
        print(f'[SI2026] CHECKPOINT PASS: {Path(path).name}, actor=35, critic=35', flush=True)
    runner.save = checked_save
