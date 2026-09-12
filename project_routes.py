"""Project-wide routes. Each process starts with the matching source family."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
# name -> (relative entry, description). Baselines are run as packages.
ENTRIES = {
 'data.materialize': ('tools/materialize_clips_v1.py','Materialize synchronized video clips'),
 'data.events': ('tools/materialize_event50h_clips_v1.py','Materialize aligned event clips'),
 'data.align': ('tools/build_memory_dense_aligned_cache_v0.py','Join source, dense and latent caches'),
 'data.manifest': ('tools/build_v2_map_manifest_and_cache_v1.py','Build training map/cache manifests'),
 'data.multiview': ('tools/build_multiview_consistency_manifest_v1.py','Build same-scene multiview groups'),
 'data.inspect-multiview': ('tools/multiview_consistency_data_v1.py','Validate multiview group data'),
 'data.visible-pairs': ('tools/find_mutual_visible_clip_pairs_v0.py','Find mutually visible clip pairs'),
 'geometry.static': ('tools/build_static_memory_v0.py','Build static map memory'),
 'geometry.episode': ('tools/build_episode_memory_v0.py','Build dynamic episode memory'),
 'geometry.render': ('tools/run_dense_v2_gpu_v1.py','Render dense v2 on GPU'),
 'geometry.mesh': ('tools/build_mesh_dense_condition_v0.py','Render mesh-based conditions'),
 'geometry.compare': ('tools/compare_geometry_backends_v0.py','Compare geometry backends'),
 'conditions.state': ('tools/build_state_channels_v2.py','Build state v2 channels'),
 'conditions.interaction': ('tools/build_interaction_sidecar_v0.py','Build interaction sidecars'),
 'conditions.actions': ('tools/build_interaction_action_cache_v0.py','Build action cache'),
 'train.dense': ('tools/train_memory_dense_adapter_v0.py','Dense adapter training'),
 'train.base': ('tools/train_memory_dense_adapter_base_v0.py','Base-expert dense training'),
 'train.state': ('tools/train_memory_dense_adapter_state_v0.py','Dense + state training'),
 'train.interaction': ('tools/train_memory_dense_adapter_interaction_v1.py','Dense + state + interaction training'),
 'infer.dense': ('tools/sample_memory_dense_adapter_phase2a_v0.py','Dense-conditioned sampling'),
 'infer.state': ('tools/sample_memory_dense_adapter_state_v0.py','Dense + state sampling'),
 'infer.interaction': ('tools/sample_memory_dense_adapter_interaction_eval_v1.py','Interaction/control sampling'),
 'infer.ar': ('tools/qxq_sample_candidate1_ar_v0.py','Sequential autoregressive video windows'),
 'eval.loss': ('tools/evaluate_memory_dense_adapter_base_v1.py','Validation diffusion loss'),
 'eval.subsets': ('tools/build_interaction_eval_subsets_v1.py','Construct held-out evaluation subsets'),
 'eval.supervision': ('tools/audit_interaction_supervision_v0.py','Audit supervision alignment'),
 'eval.symmvc': ('tools/symmvc_v2_score_haware_v1.py','Homography-aware multiview consistency'),
 'eval.ablation-rows': ('tools/analyze_ablation_eval_rows_v0.py','Summarize ablation rows'),
 'baseline.lingbot-train': ('baselines/lingbot_cam/train.py','Camera-only LingBot LoRA'),
 'baseline.lingbot-infer': ('baselines/lingbot_cam/infer.py','Camera-only LingBot inference'),
 'baseline.lingbot-validate': ('baselines/lingbot_cam/validate_manifest.py','Validate LingBot baseline data'),
 'baseline.scope-train': ('baselines/scope/train.py','SCOPE ActionModule fine-tuning'),
 'baseline.scope-infer': ('baselines/scope/infer.py','SCOPE zero-shot / fine-tuned inference'),
 'baseline.scope-validate': ('baselines/scope/validate_manifest.py','Validate SCOPE baseline data'),
 'baseline.scope-actions': ('baselines/scope/actions/convert_csgo_to_scope.py','Convert CSGO actions to SCOPE format'),
}
SCENE_ENTRIES = {
 'data.latents': ('cache_tier69h_latents_dlc_v0.py','Cache LingBot video/text latents'),
 'data.covis': ('mine_scene_covis_pairs_v3.py','Mine scene co-visibility pairs'),
 'data.heldout-pairs': ('mine_strong_covis_pairs_heldout_v1.py','Mine held-out strong co-visibility'),
 'dynamics.train-v0': ('train_player_dynamics_v0.py','Player dynamics v0'),
 'dynamics.train-v1': ('train_player_dynamics_v1.py','Player dynamics v1'),
 'dynamics.train-v2': ('train_player_dynamics_v2.py','Player dynamics v2'),
 'dynamics.train-v3': ('train_player_dynamics_v3_rollout_v0.py','Scheduled multistep dynamics training'),
 'dynamics.fit-motion': ('fit_motion_params_v1.py','Fit map-aware motion parameters'),
 'dynamics.evaluate': ('evaluate_player_dynamics_v2_long_horizon_v0.py','Long-horizon dynamics evaluation'),
 'rollout.dynamics': ('run_dynamics_inference_demo_v0.py','Map-aware dynamics to multiview generation'),
 'rollout.multiscene': ('run_multiscene_10ego_demo_v1.py','Ten-ego two-window scene pipeline'),
 'rollout.native': ('run_native_dynamics_10view_10s_v1.py','Native dynamics ten-view 10-second pipeline'),
 'rollout.srcds': ('run_srcds_inference_rollout_v1.py','SRCDS engine-backed inference rollout'),
 'eval.structure': ('structure_adherence_score_v0.py','Condition structure adherence'),
 'eval.zero-visibility': ('eval_zero_visibility_hallucination_v0.py','Zero-visibility hallucination evaluation'),
 'eval.ar-drift': ('ar_drift_curve_v0.py','Autoregressive temporal drift'),
 'eval.significance': ('paired_significance_v0.py','Paired significance analysis'),
 'infer.history-guidance': ('qxq_sample_candidate1_ar_hg_v0.py','AR sampling with history guidance'),
}
ENTRIES.update({k:(f'pipelines/scene_rollout/tools/{p}',d) for k,(p,d) in SCENE_ENTRIES.items()})


def route_command(name: str, args: list[str]):
    if name not in ENTRIES:
        raise ValueError(f'Unknown route {name!r}; run multiview.py list')
    relative, _ = ENTRIES[name]
    entry = ROOT / relative
    family = ROOT / 'pipelines/scene_rollout' if relative.startswith('pipelines/') else ROOT
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join(dict.fromkeys(map(str,[family/'tools',ROOT,ROOT/'tools',ROOT/'vendor/lingbot'])))
    argv = [sys.executable]
    if relative.startswith('baselines/'):
        argv += ['-m', relative[:-3].replace('/','.')]
    else:
        argv.append(str(entry))
    return argv + args, family, env


def main(arguments: list[str]) -> int:
    if arguments == ['list']:
        for name,(_,desc) in ENTRIES.items(): print(f'{name:28} {desc}')
        return 0
    parser = argparse.ArgumentParser(description='Run a named project route; arguments after -- go directly to that tool.')
    parser.add_argument('action', choices=['run'])
    parser.add_argument('route', choices=sorted(ENTRIES))
    parser.add_argument('--dry-run',action='store_true')
    # Keep tool --help/subcommands untouched. Wrapper options precede --.
    before, sep, after = '\0'.join(arguments).partition('\0--\0')
    wrapper = before.split('\0') if before else []
    forwarded = after.split('\0') if sep else []
    ns = parser.parse_args(wrapper)
    argv,cwd,env = route_command(ns.route,forwarded)
    print(shlex.join(argv),flush=True)
    if ns.dry_run: return 0
    return subprocess.call(argv,cwd=cwd,env=env)
