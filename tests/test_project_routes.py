import os
from pathlib import Path
import subprocess
import sys

from project_routes import ROOT, route_command
from runtime_paths import source_path


def test_scene_route_keeps_its_flat_dependencies_first():
    argv,cwd,env=route_command('rollout.multiscene',['prepare','--help'])
    assert cwd==ROOT/'pipelines/scene_rollout'
    assert env['PYTHONPATH'].split(os.pathsep)[0]==str(cwd/'tools')
    assert argv[-2:]==['prepare','--help']


def test_baseline_runs_as_package_and_portable_code_roots():
    argv,cwd,env=route_command('baseline.lingbot-train',['--help'])
    assert argv[1:3]==['-m','baselines.lingbot_cam.train']
    assert cwd==ROOT
    assert source_path('scene','tools/run_multiscene_10ego_demo_v1.py').is_file()
    assert source_path('lingbot','wan/modules/model.py').is_file()


def test_route_dry_run_forwards_subcommand_and_paths_without_side_effects(tmp_path):
    out=tmp_path/'never create; literal'
    p=subprocess.run([sys.executable,str(ROOT/'multiview.py'),'run','rollout.multiscene','--dry-run','--','prepare','--output-root',str(out)],text=True,capture_output=True,cwd=tmp_path)
    assert p.returncode==0,p.stderr
    assert 'prepare' in p.stdout and str(out) in p.stdout
    assert not out.exists()
