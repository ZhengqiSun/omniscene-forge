#!/usr/bin/env python3
"""Import the complete project source selection into an empty directory.

Families retain their own flat-module dependency versions. Source selection is
recorded for every input file; one-off jobs and generated artifacts stay in the
original backup. This creates the pre-portability-patch source snapshot.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

QXQ = 'xiaoqi/multiview-map-v0-qxq'
SCENE = 'zhengqi/multiview-map-v0/code_interaction_v1_20260731_v0'
# Fixed experiment gates/watchers/launchers carry machine/job-specific state.
OPS = re.compile(r'(?:^|_)(?:w[2-7]|watcher|watch|dlc_submit|gpu_hold|probe_v2_path)(?:_|\.)|^(?:package_|print_|write_|launch_|dlc_import|scan_tier|run_lingbot_fast_v2_|audit_lingbot_fast)')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def closure(root, seeds):
    candidates = {p.relative_to(root).as_posix(): p for p in root.rglob('*.py') if 'yitao' not in str(p.relative_to(root)) and 'hongfeng' not in str(p.relative_to(root))}
    by_name = {}
    for path in candidates.values():
        by_name.setdefault(path.name, []).append(path)
    pending, selected, edges = list(seeds), set(), {}
    while pending:
        path = pending.pop()
        if path in selected:
            continue
        selected.add(path)
        text = path.read_text(encoding="utf-8-sig")
        tree = ast.parse(text, str(path))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(a.name.split('.')[0] + '.py' for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module.split('.')[0] + '.py')
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                # Covers importlib file loading and subprocess script arguments.
                names.update(re.findall(r'\b([a-zA-Z_][\w]*\.py)\b', node.value))
                if node.value + '.py' in by_name:
                    names.add(node.value + '.py')
        deps = set()
        for name in names:
            choices = by_name.get(name, [])
            if choices:
                deps.add(next((p for p in choices if p.parent == path.parent), next((p for p in choices if p.parent == root), choices[0])))
        edges[path] = deps
        pending.extend(deps - selected)
    return selected, edges

def assemble(source, output):
    source, output = source.resolve(), output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output must be empty')
    output.mkdir(parents=True, exist_ok=True)
    mapping, dependencies = {}, {}
    for lineage, target in [(QXQ, Path('tools')), (SCENE, Path('pipelines/scene_rollout/tools'))]:
        root = source / lineage / 'tools'
        seeds = [p for p in root.glob('*.py') if not OPS.search(p.name)]
        selected, edges = closure(root, seeds)
        for p in selected:
            mapping[p] = target / p.relative_to(root)
        # Small co-located fixtures are runtime resources, not model/data caches.
        for p in root.glob("*.json"):
            mapping[p] = target / p.name
        dependencies.update({str(p.relative_to(source)): sorted(str(d.relative_to(source)) for d in deps) for p,deps in edges.items()})
    # Baseline implementations, their contracts and their real unit tests.
    for p in (source / QXQ / 'baselines').rglob('*'):
        if p.is_file() and (p.suffix == '.py' or p.name == 'SCOPE_INTERFACE_SPEC.md'):
            mapping[p] = Path('baselines') / p.relative_to(source / QXQ / 'baselines')
    # Keep the complete uploaded LingBot runtime, plus its public generators.
    vendor = source / 'xiaoqi/lingbot-world'
    for p in vendor.rglob('*.py'):
        if p.is_relative_to(vendor / 'wan') or p.parent == vendor:
            mapping[p] = Path('vendor/lingbot') / p.relative_to(vendor)
    mapping[vendor / 'LICENSE.txt'] = Path('vendor/lingbot/LICENSE.txt')
    records = []
    for p, rel in sorted(mapping.items()):
        dest = output / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
        records.append({'path':str(rel), 'source_path':str(p.relative_to(source)), 'source_sha256':sha(p)})
    inventory = []
    for lineage in [QXQ, SCENE]:
        for p in sorted((source / lineage).rglob('*')):
            if not p.is_file():
                continue
            if p in mapping:
                reason, target = 'included', str(mapping[p])
            else:
                target = None
                rel = str(p.relative_to(source / lineage))
                reason = ('user-excluded legacy project' if any(n in rel.lower() for n in ['yitao','hongfeng']) else
                          'unreferenced legacy module' if '/legacy/' in rel else
                          'fixed experiment orchestration / presentation / scratch' if p.suffix in {'.py','.sh'} else
                          'historical documentation, configuration, or generated artifact')
            inventory.append({'source_path':str(p.relative_to(source)), 'disposition':reason, 'target':target})
    manifest = {'repository':'https://github.com/ZhengqiSun/multi-views-lingbot',
                'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip(),
                'source_branch':'runtime-complete-20260912',
                'selected_lineages': {'tools':QXQ,'pipelines/scene_rollout/tools':SCENE,'vendor/lingbot':'xiaoqi/lingbot-world'},
                'files':records,'source_dependency_edges':dependencies}
    (output/'provenance').mkdir(exist_ok=True)
    (output/'provenance/source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (output/'provenance/selection_inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
    print(json.dumps({'imported_files':len(records), 'source_files_classified':len(inventory)},indent=2))

if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source',type=Path); ap.add_argument('output',type=Path)
    args=ap.parse_args(); assemble(args.source,args.output)
