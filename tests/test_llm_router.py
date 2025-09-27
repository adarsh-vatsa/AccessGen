import os
import sys
from pathlib import Path
import json
import types
import importlib

# Ensure repository root is on sys.path so that `src` package is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def install_module(name: str, module: types.ModuleType):
    parts = name.split('.')
    # Create parent packages chain
    for i in range(1, len(parts)):
        parent_name = '.'.join(parts[:i])
        child_name = '.'.join(parts[:i+1])
        if parent_name not in sys.modules:
            sys.modules[parent_name] = types.ModuleType(parent_name)
        if child_name not in sys.modules:
            sys.modules[child_name] = types.ModuleType(child_name)
        setattr(sys.modules[parent_name], parts[i], sys.modules[child_name])
    # Install the final module object and bind to its parent
    sys.modules[name] = module
    if len(parts) > 1:
        parent = sys.modules['.'.join(parts[:-1])]
        setattr(parent, parts[-1], module)


def clear_modules(prefixes):
    for k in list(sys.modules.keys()):
        if any(k == p or k.startswith(p + '.') for p in prefixes):
            sys.modules.pop(k, None)


def fresh_openai_module():
    if 'src.llm.openai_router' in sys.modules:
        importlib.reload(sys.modules['src.llm.openai_router'])
        return sys.modules['src.llm.openai_router']
    return importlib.import_module('src.llm.openai_router')


def test_generate_openai_text(monkeypatch):
    clear_modules(['openai'])

    # Install our script module so requests.getmodule finds it
    install_module('openai_router', types.ModuleType('openai_router'))

    class FakeResp:
        def __init__(self, text):
            self.output_text = text

    class FakeResponses:
        def __init__(self):
            self.calls = []
        def create(self, **kwargs):
            self.calls.append(kwargs)
            return FakeResp('ok-openai')

    class FakeOpenAI:
        def __init__(self):
            self.responses = FakeResponses()

    fake_request_mod = types.ModuleType('requests')

    def fake_post(url, headers=None, data=None, timeout=None):
        payload = json.loads(data)
        assert payload["model"] == 'gpt-5'
        fake_openai_module = types.ModuleType('openai')
        fake_openai_module.OpenAI = FakeOpenAI
        install_module('openai', fake_openai_module)
        resp = FakeResp('ok-openai')
        resp.json = lambda: {"output_text": 'ok-openai'}
        resp.status_code = 200
        return resp

    fake_request_mod.post = fake_post
    install_module('requests', fake_request_mod)

    monkeypatch.setenv('OPENAI_API_KEY', 'test-key')

    mod = fresh_openai_module()
    out = mod.generate_openai_text('hello', model='gpt-5', max_output_tokens=8)
    assert out == 'ok-openai'


def test_generate_openai_json(monkeypatch):
    clear_modules(['openai'])
    install_module('openai_router', types.ModuleType('openai_router'))

    def fake_json_response():
        return {
            "output": [],
            "output_text": json.dumps({"ok": True}),
        }

    class FakeResp:
        def __init__(self):
            self.status_code = 200
        def json(self):
            return fake_json_response()

    fake_request_mod = types.ModuleType('requests')

    def fake_post(url, headers=None, data=None, timeout=None):
        return FakeResp()

    fake_request_mod.post = fake_post
    install_module('requests', fake_request_mod)

    monkeypatch.setenv('OPENAI_API_KEY', 'test-key')

    mod = fresh_openai_module()
    out = mod.generate_openai_json('ping', model='gpt-5', schema={"title": "s"}, max_output_tokens=32)
    assert out == {"ok": True}
# Ensure repository root is on sys.path so that `src` package is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
