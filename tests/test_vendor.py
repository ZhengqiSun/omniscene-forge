import inspect
from pathlib import Path

import wan
from wan.modules.model import WanModel
from multiview import ROOT


def test_import_uses_bundled_wan_and_adapter_interface():
    assert Path(wan.__file__).resolve().is_relative_to(ROOT / 'vendor/lingbot')
    assert callable(wan.WanI2V) and callable(wan.WanI2VFast)
    assert 'dit_cond_dict' in inspect.signature(WanModel.forward).parameters
