"""Independent, non-training evaluation of the verified SI2026 checkpoint.

Logs every 5 ms physics step in all environments; no inherited logger is used.
Only evaluation target sequences and state resets differ from the saved plant.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--num_envs', type=int, default=32)
parser.add_argument('--seed', type=int, default=20261002)
parser.add_argument('--bpms', type=int, nargs='+', default=[60, 80])
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
app = launcher.app

import csv
import inspect
import math
import time
from datetime import datetime, timezone
import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab.utils.io import dump_yaml
import isaaclab_tasks
import porcaro_2026.tasks
import porcaro_2026

TASK = 'Template-Porcaro-2026-ModelB-DR-user1'
PACKAGE = Path('/home/labuser1/IsaacLab/source/porcaro_2026/porcaro_2026')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p, obj): Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
def require(test, message):
    if not test: raise RuntimeError(message)

def main():
    out = Path(args.output).resolve()
    require(not out.exists(), 'Use a new output directory; no evaluation is overwritten.')
    out.mkdir(parents=True)
    ckpt = Path(args.checkpoint).resolve()
    source_report = json.loads((ckpt.parent / 'verified_runtime.json').read_text())
    require(Path(porcaro_2026.__file__).resolve().parent == PACKAGE, 'Wrong package mapping')
    hashes = source_report['source_sha256']
    for rel, value in hashes.items():
        if '/tools/' not in rel:
            require(sha(PACKAGE / rel) == value, 'Training source changed: ' + rel)
    for path, value in source_report['asset_sha256'].items():
        require(sha(path) == value, 'Training asset changed: ' + path)

    cfg = load_cfg_from_registry(TASK, 'env_cfg_entry_point')
    acfg = load_cfg_from_registry(TASK, 'rsl_rl_cfg_entry_point')
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.sim.device = args.device or 'cuda:0'
    cfg.logging.enabled = False
    acfg.device = cfg.sim.device
    require(cfg.lookahead_horizon == 0.5 and cfg.observation_space == 35, 'Wrong observation cfg')
    require(cfg.controller.Pmax == 0.6 and cfg.controller.discrete_threshold == 0.5, 'Wrong binary cfg')
    env = RslRlVecEnvWrapper(gym.make(TASK, cfg=cfg), clip_actions=acfg.clip_actions)
    plant = env.unwrapped
    runner = OnPolicyRunner(env, acfg.to_dict(), log_dir=None, device=acfg.device)
    payload = torch.load(ckpt, weights_only=True, map_location=acfg.device)
    shape = list(payload['model_state_dict']['memory_a.rnn.weight_ih_l0'].shape)
    require(shape == [512, 35], 'Wrong checkpoint LSTM shape')
    runner.alg.policy.load_state_dict(payload['model_state_dict'])
    inference = runner.get_inference_policy(device=acfg.device)
    require(plant.lookahead_steps == 25, 'Wrong runtime lookahead')
    require(type(plant.action_controller).__name__ == 'DiscreteTorqueActionController', 'Wrong controller')
    dump_yaml(str(out / 'evaluation_env.yaml'), cfg)
    dump_yaml(str(out / 'evaluation_agent.yaml'), acfg)

    protocol = {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'task': TASK,
        'checkpoint': str(ckpt), 'checkpoint_sha256': sha(ckpt), 'checkpoint_iteration': payload.get('iter'),
        'evaluation_script_sha256': sha(__file__), 'package': str(PACKAGE), 'source_matches_training': True,
        'lstm_weight_shape': shape, 'observation_dimension': 35, 'lookahead_s': 0.5,
        'lookahead_samples': 25, 'physics_dt_s': cfg.sim.dt, 'control_dt_s': plant.dt_ctrl_step,
        'seed': args.seed, 'num_environments': args.num_envs, 'bpms': args.bpms,
        'policy': 'deterministic inference; no parameter updates',
        'evaluation_design': 'One 4-bar episode per environment per condition; first bar silent; last 3 bars scored.',
        'replication': 'Parallel randomized instances of one trained policy, not independent training seeds.',
        'initialization': 'Robot default state; pressure controller and policy LSTM explicitly reset before each condition.',
        'training_pattern_offsets_on_16th_grid': dict(plant.rhythm_generator.rudiments),
        'evaluation_patterns': {'quarter': [0,4,8,12], 'eighth': [0,2,4,6,8,10,12,14]},
        'force_definition': 'Scaled contact resultant = 3 * norm(raw net_forces_w); simulation quantity, not measured physical force.',
        'force_multiplier': float(plant.force_scale_sim_to_real),
        'event_definition': 'First upward crossing above 1 N; merge threshold gaps shorter than 10 ms; ignore crossings before first target bar.',
        'matching': 'One-to-one nearest target within +/- 6/BPM s; target times quantized to 20 ms control grid.',
        'event_peak': 'Maximum scaled resultant in [contact onset, onset + 6/BPM s).',
        'row_timing': 'time_s is end of physics step. control_index and target_force_before_N refer to pre-step index; reward uses post-control index.',
        'conditions': [],
    }
    save(out / 'provenance.json', protocol)
    original_update = plant.scene.update
    capture = {'enabled': False, 'tick': 0, 'writer': None, 'buffer': [], 'condition': None}
    columns = ['condition','env_id','physics_step','control_index','time_s','bpm',
               'target_force_before_N','raw_fx_N','raw_fy_N','raw_fz_N','scaled_force_norm_N',
               'q_wrist_rad','q_grip_rad','qd_wrist_rad_s','qd_grip_rad_s',
               'action_DF','action_F','action_G','Pcmd_DF_MPa','Pcmd_F_MPa','Pcmd_G_MPa',
               'Pout_DF_MPa','Pout_F_MPa','Pout_G_MPa']
    def update(dt, *a, **kw):
        original_update(dt, *a, **kw)
        if not capture['enabled']: return
        capture['tick'] += 1
        force = plant.stick_sensor.data.net_forces_w
        if force.ndim == 3: force = force[:, 0, :]
        q, qd = plant._get_corrected_joint_state()
        tel = plant.action_controller.get_last_telemetry()
        vals = torch.cat((plant.rhythm_generator.get_current_target(plant.episode_length_buf)[:,None],
                          force, torch.linalg.vector_norm(force,dim=-1)[:,None] * plant.force_scale_sim_to_real,
                          q, qd, plant.actions, tel['P_cmd'], tel['P_out']), dim=1).detach().cpu().numpy()
        require(np.isfinite(vals).all(), 'Non-finite raw physical trajectory')
        p = vals[:,12:15]
        require(np.logical_or(np.isclose(p,0,atol=1e-6),np.isclose(p,.6,atol=1e-6)).all(), 'Non-binary pressure command')
        tick = capture['tick']
        c = capture['condition']
        idx = (tick-1)//cfg.decimation
        for i, row in enumerate(vals):
            capture['buffer'].append([c['name'],i,tick,idx,round(tick*cfg.sim.dt,8),c['bpm']] + row.tolist())
        if len(capture['buffer']) >= 4096:
            capture['writer'].writerows(capture['buffer'])
            capture['buffer'].clear()
    plant.scene.update = update

    gen = plant.rhythm_generator
    gen.rudiments['single_4'] = [0,4,8,12]
    gen.rudiments['single_8'] = [0,2,4,6,8,10,12,14]
    with (out/'target_events.csv').open('w',newline='') as target_file:
        tw = csv.writer(target_file)
        tw.writerow(['condition','env_id','target_id','control_index','time_s','target_peak_N'])
        for bpm in args.bpms:
            for label, key in [('quarter','single_4'),('eighth','single_8')]:
                name = f'{label}_{bpm}bpm'
                gen.set_test_mode(True,float(bpm),key)
                capture['enabled'] = False
                print('[SI2026 EVAL] RESET '+name,flush=True)
                env.reset()
                plant.action_controller.reset(args.num_envs, plant.device)
                plant.actions.zero_()
                plant.prev_actions.zero_()
                runner.alg.policy.reset(torch.ones(args.num_envs,device=plant.device,dtype=torch.bool))
                obs = env.get_observations()
                require(plant._get_observations()['policy'].shape == (args.num_envs,35), 'Runtime obs not 35D')
                require(bool(torch.isfinite(plant._get_observations()['policy']).all()), 'Nonfinite initial obs')
                steps = int(plant.episode_duration_steps[0])
                offsets = gen.rudiments[key]
                # Float32 arithmetic matches RhythmGenerator.reset exactly.
                unit = torch.tensor(15.0/float(bpm)/plant.dt_ctrl_step,dtype=torch.float32)
                centers = [int(torch.round(bar*16*unit + off*unit)) for bar in (1,2,3) for off in offsets]
                actual = torch.nonzero(torch.isclose(gen.target_trajectories[0], torch.tensor(plant.target_hit_force,device=plant.device)),as_tuple=False).flatten().cpu().tolist()
                require(actual == centers, f'Target pattern mismatch: {actual} vs {centers}')
                for i in range(args.num_envs):
                    for j,t in enumerate(centers): tw.writerow([name,i,j,t,t*plant.dt_ctrl_step,plant.target_hit_force])
                target_file.flush()
                c = {'name':name,'note':label,'bpm':bpm,'control_steps':steps,'duration_s':steps*plant.dt_ctrl_step,
                     'scoring_start_s':centers[0]*plant.dt_ctrl_step,'targets_per_environment':len(centers)}
                capture.update(tick=0,condition=c)
                start = time.monotonic()
                with (out/f'{name}.csv').open('w',newline='') as f:
                    capture['writer'] = csv.writer(f)
                    capture['writer'].writerow(columns)
                    capture['enabled'] = True
                    with torch.no_grad():
                        for step in range(steps):
                            action = inference(obs)
                            obs, rew, dones, extra = env.step(action)
                            if step < steps-1: require(not bool(dones.any()), 'Unexpected early termination')
                            if (step+1)%200 == 0: print(f'[SI2026 EVAL] {name} {step+1}/{steps}',flush=True)
                    capture['enabled'] = False
                    capture['writer'].writerows(capture['buffer'])
                    capture['buffer'].clear()
                c['physics_steps'] = capture['tick']
                c['wall_seconds'] = time.monotonic()-start
                c['csv_sha256'] = sha(out/f'{name}.csv')
                protocol['conditions'].append(c)
                save(out/'provenance.json',protocol)
                print(f'[SI2026 EVAL] COMPLETE {name}: {capture["tick"]*args.num_envs} rows',flush=True)
    protocol['completed_utc'] = datetime.now(timezone.utc).isoformat()
    protocol['target_events_sha256'] = sha(out/'target_events.csv')
    protocol['status'] = 'completed'
    save(out/'provenance.json',protocol)
    env.close()
    print('[SI2026 EVAL] ALL COMPLETE '+str(out),flush=True)

if __name__ == '__main__':
    try: main()
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally: app.close()
