#!/usr/bin/env python3
"""Check syntax, imported source inventory, and optional public CLI imports."""
from __future__ import annotations
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from project_routes import ENTRIES, route_command


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--help-smoke',action='store_true')
    ap.add_argument('--out',type=Path)
    args=ap.parse_args()
    files=[p for folder in ['tools','pipelines','baselines','vendor','scripts','tests'] for p in (ROOT/folder).rglob('*.py')]
    files.extend(ROOT.glob('*.py'))
    failures=[]
    for p in files:
        try: ast.parse(p.read_bytes(),str(p))
        except SyntaxError as e: failures.append({'path':str(p.relative_to(ROOT)),'error':str(e)})
    source=json.loads((ROOT/'provenance/source_manifest.json').read_text())
    absent=[r['path'] for r in source['files'] if not (ROOT/r['path']).is_file()]
    report={'python_files':len(files),'syntax_errors':failures,'missing_imported_sources':absent,'routes':len(ENTRIES)}
    if args.help_smoke:
        def check(name):
            argv,cwd,env=route_command(name,['--help'])
            try:
                p=subprocess.run(argv,cwd=cwd,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60)
                code = p.returncode if p.returncode else (0 if 'usage:' in p.stdout.lower() else 65)
                return {'route':name,'exit_code':code,'output_tail':p.stdout[-2200:] if code else ''}
            except subprocess.TimeoutExpired:
                return {'route':name,'exit_code':124,'output_tail':'Timeout after 60 seconds'}
        with ThreadPoolExecutor(max_workers=4) as pool: report['cli_help']=list(pool.map(check,ENTRIES))
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return int(bool(failures or absent or any(r['exit_code'] for r in report.get('cli_help',[]))))

if __name__=='__main__': raise SystemExit(main())
