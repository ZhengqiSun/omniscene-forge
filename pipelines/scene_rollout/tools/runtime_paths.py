"""Portable locations for code and external research assets."""
import os
from pathlib import Path

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'multiview.py').is_file())
ASSET_ROOT = Path(os.environ.get('MULTIVIEW_ASSETS', REPO_ROOT / 'assets')).expanduser().resolve()
LINGBOT_ROOT = REPO_ROOT / 'vendor/lingbot'
SCENE_ROOT = REPO_ROOT / 'pipelines/scene_rollout'


def source_path(anchor: str, relative: str = '') -> Path:
    roots = {'project': REPO_ROOT, 'scene': SCENE_ROOT, 'lingbot': LINGBOT_ROOT,
             'assets': ASSET_ROOT, 'external': ASSET_ROOT / 'external'}
    return roots[anchor] / relative
