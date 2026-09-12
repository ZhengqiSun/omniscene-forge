#!/usr/bin/env python3
"""Replace reviewed old server path literals; preserve all algorithm source text.

Only standalone string literals with recognized prefixes are transformed. Raw
asset files and historical f-string snippets are not silently rewritten.
"""
import ast
from pathlib import Path

PREFIXES = [
 ("/mnt/data/pku/zhengqi/models/lingbot-world-v2-code","assets","external/lingbot-world-v2-code"),
 ("/mnt/data/pku/zhengqi/models/lingbot-world-v2-14b-causal-fast","assets","lingbot-world-v2-14b-causal-fast"),
 ("/mnt/data/csgo-datasets-fullsubset","assets","csgo-datasets-fullsubset"),
 ("/mnt/data/pku/xiaoqi/datasets","assets","datasets"),
 ("/mnt/workspace/xiaoqi/datasets","assets","datasets"),
 ('/mnt/workspace/zhengqi/multiview-map-v0/code_interaction_v1_20260731_v0','scene',''),
 ('/mnt/data/pku/zhengqi/multiview-map-v0/code_interaction_v1_20260731_v0','scene',''),
 ('/mnt/data/pku/zhengqi/multiview-map-v0','scene',''),
 ('/mnt/workspace/zhengqi/multiview-map-v0','scene',''),
 ('/mnt/data/pku/xiaoqi/multigen/multiview-map-v0-qxq','project',''),
 ('/mnt/workspace/xiaoqi/multigen/multiview-map-v0-qxq','project',''),
 ('/mnt/data/pku/xiaoqi/lingbot-world','lingbot',''),
 ('/mnt/workspace/xiaoqi/lingbot-world','lingbot',''),
 ('/mnt/data/pku/datasets/robbyant/lingbot-world-base-cam','assets','lingbot-world-base-cam'),
 ('/mnt/workspace/datasets/robbyant/lingbot-world-base-cam','assets','lingbot-world-base-cam'),
 ('/mnt/data/csgo-datasets','assets','csgo-datasets'),
 ('/mnt/workspace/xiaoqi/multigen/s4_train_inputs_v2_20260731','assets','prepared'),
]

def port(path):
    data = path.read_bytes()
    tree = ast.parse(data)
    parents = {c:n for n in ast.walk(tree) for c in ast.iter_child_nodes(n)}
    # AST columns are UTF-8 byte offsets. Normalize only the BOM before offsets.
    data = data.removeprefix(b'\xef\xbb\xbf')
    lines = data.splitlines(keepends=True)
    starts = [0]
    for line in lines: starts.append(starts[-1]+len(line))
    edits = []
    for node in ast.walk(tree):
        if not isinstance(node,ast.Constant) or not isinstance(node.value,str) or isinstance(parents.get(node),(ast.JoinedStr, ast.Expr)):
            continue
        value = node.value
        expression = None
        if value == '/root/bin/ffmpeg': expression = repr('ffmpeg')
        for prefix,anchor,base in PREFIXES:
            if value == prefix or value.startswith(prefix+'/'):
                relative = '/'.join(filter(None,[base,value[len(prefix):].lstrip('/')]))
                if '\n' not in relative:
                    expression = f'str(source_path({anchor!r}, {relative!r}))'
                break
        if expression:
            edits.append((starts[node.lineno-1]+node.col_offset,starts[node.end_lineno-1]+node.end_col_offset,expression.encode()))
    if not edits: return 0
    for start,end,replacement in sorted(edits,reverse=True): data=data[:start]+replacement+data[end:]
    text=data.decode()
    tree=ast.parse(text)
    first=0
    for node in tree.body:
        if isinstance(node,ast.Expr) and isinstance(node.value,ast.Constant) and isinstance(node.value.value,str): first=node.end_lineno
        elif isinstance(node,ast.ImportFrom) and node.module=='__future__': first=node.end_lineno
        else: break
    lines=text.splitlines(keepends=True)
    module='tools.runtime_paths' if path.parts[0]=='baselines' else 'runtime_paths'
    lines.insert(first,f'from {module} import source_path\n')
    path.write_text(''.join(lines))
    return len(edits)

if __name__=='__main__':
    count=0
    for root in ['tools','pipelines/scene_rollout/tools','baselines']:
        for p in Path(root).rglob('*.py'):
            if p.name!='runtime_paths.py': count+=port(p)
    print(f'Ported {count} path literals')
