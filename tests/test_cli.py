import json
from pathlib import Path

import pytest

from multiview import ROOT, build_command, option_schema, COMMANDS


def config(tmp_path, name='infer', **changes):
    data = json.loads((ROOT / 'configs' / 'interaction' / f'{name}.example.json').read_text())
    data['args'].update(changes)
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(data))
    return path


def test_paths_resolve_against_config_not_cwd(tmp_path, monkeypatch):
    path = config(tmp_path, **{'out-dir': './results', 'clip-id': 'clip with spaces; literal'})
    monkeypatch.chdir('/')
    argv, resolved = build_command('infer', path)
    assert resolved['out-dir'] == str(tmp_path / 'results')
    assert argv[argv.index('--clip-id') + 1] == 'clip with spaces; literal'
    assert resolved['lingbot-repo'] == str(ROOT / 'vendor/lingbot')
    assert not (tmp_path / 'results').exists()


def test_inference_requires_checkpoint_for_interaction(tmp_path):
    with pytest.raises(ValueError, match='adapter-checkpoint-low'):
        build_command('infer', config(tmp_path, **{'adapter-checkpoint-low': None}))


def test_unknown_option_and_code_override_fail(tmp_path):
    with pytest.raises(ValueError, match='Unknown'):
        build_command('infer', config(tmp_path, typo=1))
    with pytest.raises(ValueError, match='Code paths'):
        build_command('infer', config(tmp_path, **{'lingbot-repo': '/some/other/wan'}))


def test_booleans_and_smoke_bounds(tmp_path):
    path = config(tmp_path, 'smoke', **{'max-steps': 999, 'reset-optimizer': False})
    argv, resolved = build_command('smoke', path)
    assert resolved['max-steps'] == '1'
    assert resolved['reset-optimizer'] is True
    assert '--no-require-release-report' in argv
    assert '--activation-checkpoint-adapter' in argv
    assert '-m' in argv and 'torch.distributed.run' in argv


def test_repeatable_paths_and_multi_value_options(tmp_path):
    path = config(tmp_path, 'smoke', **{'extra-map-manifest': ['a.json', 'b.json'], 'dense-hw': [240, 416]})
    argv, _ = build_command('smoke', path)
    assert argv.count('--extra-map-manifest') == 2
    pos = argv.index('--dense-hw')
    assert argv[pos + 1:pos + 3] == ['240', '416']


def test_pair_group_paths_are_resolved(tmp_path):
    path = tmp_path / 'eval.json'
    path.write_text(json.dumps({'args': {'pair': [['a','a/map.json','a/cache.jsonl'],['b','b/map.json','b/cache.jsonl']],
                                        'state-cache-manifest':'state.jsonl','out':'out.jsonl'}}))
    argv, resolved = build_command('prepare-eval', path)
    assert argv.count('--pair') == 2
    assert resolved['pair'][1] == ['b', str(tmp_path/'b/map.json'), str(tmp_path/'b/cache.jsonl')]


def test_environment_expansion_and_missing_variable(tmp_path, monkeypatch):
    monkeypatch.setenv('MV_TEST_DATA', str(tmp_path))
    _, resolved = build_command('infer', config(tmp_path, **{'cache-manifest':'${MV_TEST_DATA}/cache.jsonl'}))
    assert resolved['cache-manifest'] == str(tmp_path / 'cache.jsonl')
    monkeypatch.delenv('MV_UNSET_EXAMPLE', raising=False)
    with pytest.raises(ValueError, match='Unset'):
        build_command('infer', config(tmp_path, **{'cache-manifest':'${MV_UNSET_EXAMPLE}/cache.jsonl'}))


def test_multi_node_requires_rank_and_address(tmp_path):
    path = config(tmp_path, 'train')
    data = json.loads(path.read_text())
    data['distributed'] = {'nproc_per_node':2,'nnodes':2}
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='Multi-node'):
        build_command('train', path)
    data['distributed'].update(node_rank=1,master_addr='train-master')
    path.write_text(json.dumps(data))
    argv, _ = build_command('train', path)
    assert '--node_rank=1' in argv and '--master_addr=train-master' in argv


@pytest.mark.parametrize('name', ['train','smoke','infer','prepare-state','prepare-interaction','prepare-eval'])
def test_example_configs_only_use_declared_options(name):
    argv, _ = build_command(name, ROOT / 'configs' / 'interaction' / f'{name}.example.json')
    assert argv


def test_every_public_entry_is_present():
    for filename, _ in COMMANDS.values():
        assert option_schema(ROOT / 'tools' / filename)
