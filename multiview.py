#!/usr/bin/env python3
"""Full multiview project: list routes, run a route, or use an interaction JSON recipe."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
ALIASES = {'interaction-train':'train','interaction-infer':'infer','interaction-smoke':'smoke','interaction-preflight':'preflight'}
TRAINER = 'train_memory_dense_adapter_interaction_v1.py'
COMMANDS = {
    'train': (TRAINER, 'train'),
    'smoke': (TRAINER, 'train'),
    'preflight': (TRAINER, 'preflight-train'),
    'prepare-index': (TRAINER, 'prepare-index'),
    'infer': ('sample_memory_dense_adapter_interaction_eval_v1.py', None),
    'prepare-state': ('build_state_channels_v2.py', None),
    'prepare-interaction': ('build_interaction_sidecar_v0.py', None),
    'prepare-aligned': ('build_memory_dense_aligned_cache_v0.py', None),
    'prepare-manifest': ('build_v2_map_manifest_and_cache_v1.py', None),
    'prepare-eval': ('build_interaction_eval_subsets_v1.py', None),
    'prepare-multiview': ('build_multiview_consistency_manifest_v1.py', None),
    'inspect-multiview': ('multiview_consistency_data_v1.py', None),
    'render-dense': ('run_dense_v2_gpu_v1.py', None),
    'audit-supervision': ('audit_interaction_supervision_v0.py', None),
    'evaluate-loss': ('evaluate_memory_dense_adapter_base_v1.py', None),
}
TRAIN_COMMANDS = {'train', 'smoke', 'preflight', 'prepare-index'}
REQUIRED = {
    **{c: {'map-manifest', 'cache-manifest', 'state-cache-manifest', 'ckpt-dir', 'out-dir'} for c in TRAIN_COMMANDS},
    'infer': {'map-manifest', 'cache-manifest', 'state-cache-manifest', 'ckpt-dir', 'out-dir', 'adapter-checkpoint-low'},
    'prepare-state': {'cache-manifest', 'out-dir'},
    'prepare-interaction': {'state-manifest', 'cache-manifest', 'out-dir'},
    'prepare-aligned': {'cache-manifest', 'source-manifest', 'map-manifest', 'out-jsonl', 'out-report'},
    'prepare-manifest': {'release', 'aligned', 'out-dir', 'zhengqi-root'},
    'render-dense': {'source-manifest', 'out-dir', 'bsp-faces-npz'},
    'evaluate-loss': {'map-manifest', 'cache-manifest', 'state-cache-manifest', 'ckpt-dir', 'out-dir', 'ckpt'},
}


def option_schema(entry: Path) -> dict:
    """Read argparse declarations without importing torch or running entrypoints."""
    result = {}
    for node in ast.walk(ast.parse(entry.read_bytes())):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute) or node.func.attr != 'add_argument':
            continue
        flags = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str) and a.value.startswith('--')]
        if not flags:
            continue
        kw = {k.arg: k.value for k in node.keywords}
        action = ast.unparse(kw['action']) if 'action' in kw else ''
        spec = {'path': isinstance(kw.get('type'), ast.Name) and kw['type'].id == 'Path',
                'append': action == "'append'", 'boolean_optional': action.endswith('BooleanOptionalAction'),
                'store_true': action == "'store_true'", 'store_false': action == "'store_false'",
                'required': isinstance(kw.get('required'), ast.Constant) and kw['required'].value is True,
                'nargs': ast.literal_eval(kw['nargs']) if isinstance(kw.get('nargs'), ast.Constant) else None}
        for flag in flags:
            result[flag[2:]] = spec
    return result


def load_config(path: Path) -> dict:
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or set(data) - {'args', 'distributed'}:
        raise ValueError('Config must contain only args and optional distributed objects')
    if not isinstance(data.get('args'), dict):
        raise ValueError('Config requires an args object')
    return data


def expanded(value: str) -> str:
    value = os.path.expandvars(os.path.expanduser(value))
    if re.search(r'\$\{?[A-Za-z_][A-Za-z0-9_]*', value):
        raise ValueError(f'Unset environment variable in {value!r}')
    return value


def build_command(command: str, config_path: Path, *, overrides: dict | None = None) -> tuple[list[str], dict]:
    command = ALIASES.get(command, command)
    config_path = config_path.resolve()
    data = load_config(config_path)
    values = dict(data['args'])
    values.update(overrides or {})
    filename, mode = COMMANDS[command]
    entry = ROOT / 'tools' / filename
    schema = option_schema(entry)
    unknown = set(values) - set(schema)
    if unknown:
        raise ValueError(f'Unknown options for {command}: {sorted(unknown)}')
    if 'lingbot-repo' in values:
        raise ValueError('Code paths are managed by this repository; configure asset paths instead')
    if 'lingbot-repo' in schema:
        values['lingbot-repo'] = str(ROOT / 'vendor/lingbot')
    if command in TRAIN_COMMANDS:
        values.setdefault('use-base-model', True)
        values.setdefault('base-expert', 'low')
        values.setdefault('negative-contrast-weight', 0)
    if command == 'smoke':
        values.update({'max-steps': 1, 'save-every': 1, 'log-every': 1, 'reset-optimizer': True})
    required = REQUIRED.get(command, set()) | {k for k, v in schema.items() if v['required']}
    missing = [k for k in sorted(required) if values.get(k) is None or values.get(k) == '' or values.get(k) == []]
    if missing:
        raise ValueError(f'Missing explicit options: {missing}')
    resolved, args = {}, []
    for key, value in values.items():
        if value is None:
            continue
        spec = schema[key]
        if spec['boolean_optional'] or spec['store_true'] or spec['store_false']:
            if not isinstance(value, bool):
                raise ValueError(f'{key} must be a JSON boolean')
            if spec['boolean_optional']:
                args.append('--' + ('' if value else 'no-') + key)
            elif (spec['store_true'] and value) or (spec['store_false'] and not value):
                args.append('--' + key)
            resolved[key] = value
            continue
        if isinstance(value, bool) or isinstance(value, dict):
            raise ValueError(f'{key} expects a scalar or list')
        items = value if isinstance(value, list) else [value]
        if not items:
            continue
        if spec['append'] and spec['nargs'] is not None:
            groups = items if isinstance(items[0], list) else [items]
            normalized = []
            for group in groups:
                if isinstance(spec['nargs'], int) and len(group) != spec['nargs']:
                    raise ValueError(f'{key} requires groups of {spec["nargs"]} values')
                converted_group = []
                for i, item in enumerate(group):
                    if not isinstance(item, (str, int, float)) or isinstance(item, bool):
                        raise ValueError(f'Invalid value for {key}')
                    text = expanded(str(item))
                    # prepare-eval --pair is NAME MAP_MANIFEST CACHE_MANIFEST.
                    if spec['path'] or (key == 'pair' and i > 0):
                        text = str((config_path.parent / text).resolve())
                    converted_group.append(text)
                args.extend(['--' + key, *converted_group])
                normalized.append(converted_group)
            resolved[key] = normalized
            continue
        converted = []
        for item in items:
            if not isinstance(item, (str, int, float)) or isinstance(item, bool):
                raise ValueError(f'Invalid value for {key}')
            text = expanded(str(item))
            if spec['path']:
                path = Path(text)
                text = str((config_path.parent / path).resolve()) if not path.is_absolute() else str(path)
            converted.append(text)
        if spec['append']:
            for item in converted:
                args.extend(['--' + key, item])
        else:
            args.extend(['--' + key, *converted])
        resolved[key] = converted if isinstance(value, list) else converted[0]
    argv = [sys.executable]
    dist = data.get('distributed', {})
    if not isinstance(dist, dict) or set(dist) - {'nproc_per_node', 'nnodes', 'node_rank', 'master_addr', 'master_port'}:
        raise ValueError('Invalid distributed configuration')
    if command in {'train', 'smoke'}:
        nproc, nnodes = dist.get('nproc_per_node', 1), dist.get('nnodes', 1)
        if type(nproc) is not int or type(nnodes) is not int or nproc < 1 or nnodes < 1:
            raise ValueError('nproc_per_node and nnodes must be positive integers')
        if command == 'smoke' and (nproc != 1 or nnodes != 1):
            raise ValueError('smoke is one GPU / one step; use train for distributed runs')
        argv += ['-m', 'torch.distributed.run', f'--nproc_per_node={nproc}']
        if nnodes == 1:
            argv.append('--standalone')
        else:
            rank = dist.get('node_rank')
            if type(rank) is not int or not 0 <= rank < nnodes or not dist.get('master_addr'):
                raise ValueError('Multi-node runs need a valid node_rank and master_addr')
            argv += [f'--nnodes={nnodes}', f'--node_rank={rank}',
                     f'--master_addr={expanded(str(dist["master_addr"]))}', f'--master_port={dist.get("master_port", 29500)}']
    argv += [str(entry)] + ([mode] if mode else []) + args
    return argv, resolved


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] in {"list", "run"}:
        from project_routes import main as project_main
        return project_main(sys.argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=sorted(set(COMMANDS) | set(ALIASES)))
    parser.add_argument('--config', type=Path)
    parser.add_argument('--dry-run', action='store_true', help='Resolve and print command without executing or writing output')
    parser.add_argument('--tool-help', action='store_true', help='Show underlying entrypoint options')
    ns = parser.parse_args()
    ns.command = ALIASES.get(ns.command, ns.command)
    if ns.tool_help:
        return subprocess.call([sys.executable, str(ROOT / 'tools' / COMMANDS[ns.command][0]), '--help'], cwd=ROOT, env={**os.environ, 'PYTHONPATH': os.pathsep.join([str(ROOT/'tools'),str(ROOT),str(ROOT/'vendor/lingbot')])})
    if ns.config is None:
        parser.error('--config is required unless --tool-help is used')
    try:
        argv, resolved = build_command(ns.command, ns.config)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(shlex.join(argv), flush=True)
    if ns.dry_run:
        return 0
    # Per-process record avoids multi-node writers racing over one metadata file.
    output = resolved.get('out-dir')
    if output:
        directory = Path(output)
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        sources = sorted((ROOT / 'tools').glob('*.py')) + sorted((ROOT / 'vendor/lingbot').rglob('*.py'))
        record = {'command': argv, 'resolved_args': resolved, 'config': str(ns.config.resolve()),
                  'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
        (directory / f'launch_{stamp}_{os.getpid()}.json').write_text(json.dumps(record, indent=2) + '\n')
    env = os.environ.copy()
    # The bundled Wan is used even when a different checkout is installed.
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT / 'tools'), str(ROOT), str(ROOT / 'vendor/lingbot')])
    return subprocess.call(argv, cwd=ROOT, env=env)


if __name__ == '__main__':
    raise SystemExit(main())
