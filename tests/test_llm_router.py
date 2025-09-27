import os
import sys
from pathlib import Path
import json
import types
import importlib


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


def fresh_router_module():
    # Reload to ensure clean state between tests
    if 'src.llm.router' in sys.modules:
        importlib.reload(sys.modules['src.llm.router'])
        return sys.modules['src.llm.router']
    return importlib.import_module('src.llm.router')


def test_google_legacy_generate_text(monkeypatch):
    clear_modules(['google'])

    # Fake legacy google.generativeai
    genai_legacy = types.ModuleType('google.generativeai')

    class GenerationConfig:
        def __init__(self, **kwargs):
            pass

    class Resp:
        def __init__(self, text):
            self.text = text

    class GenerativeModel:
        def __init__(self, model):
            self.model = model
        def generate_content(self, prompt, generation_config=None):
            return Resp('ok-legacy')

    def configure(api_key=None):
        return None

    genai_legacy.GenerationConfig = GenerationConfig
    genai_legacy.GenerativeModel = GenerativeModel
    genai_legacy.configure = configure

    install_module('google.generativeai', genai_legacy)

    # Env
    monkeypatch.setenv('GENAI_FORCE_LEGACY', '1')
    monkeypatch.setenv('GEMINI_API_KEY', 'test-key')

    router_mod = fresh_router_module()
    router = router_mod.ModelRouter()
    out = router.generate_text('hello', provider='google', model='models/gemini-2.5-pro', temperature=0.0, max_tokens=8)
    assert out == 'ok-legacy'


def test_google_new_generate_text(monkeypatch):
    clear_modules(['google'])

    # Fake new google.genai and google.genai.types
    genai_pkg = types.ModuleType('google.genai')
    types_pkg = types.ModuleType('google.genai.types')

    class GenerateContentConfig:
        def __init__(self, **kwargs):
            pass

    class Part:
        def __init__(self, text):
            self.text = text

    class Content:
        def __init__(self, parts):
            self.parts = parts

    class Candidate:
        def __init__(self, content):
            self.content = content

    class Resp:
        def __init__(self, text):
            # Provide via candidates/parts for extractor path
            self.candidates = [Candidate(Content([Part(text)]))]

    class Models:
        def generate_content(self, model, contents, config=None):
            return Resp('ok-new')

    class Client:
        def __init__(self, api_key=None):
            self.models = Models()

    genai_pkg.Client = Client
    types_pkg.GenerateContentConfig = GenerateContentConfig

    install_module('google.genai', genai_pkg)
    install_module('google.genai.types', types_pkg)

    monkeypatch.setenv('GENAI_FORCE_LEGACY', '0')
    monkeypatch.setenv('GEMINI_API_KEY', 'test-key')

    router_mod = fresh_router_module()
    router = router_mod.ModelRouter()
    out = router.generate_text('hello', provider='google', model='models/gemini-2.5-pro', temperature=0.0, max_tokens=8)
    assert out == 'ok-new'


def test_anthropic_generate_text(monkeypatch):
    clear_modules(['anthropic'])

    anth = types.ModuleType('anthropic')

    class Block:
        def __init__(self, text):
            self.text = text

    class Resp:
        def __init__(self):
            self.content = [Block('ok-claude')]

    class Messages:
        def create(self, **kwargs):
            return Resp()

    class Anthropic:
        def __init__(self, api_key=None):
            self.messages = Messages()

    anth.Anthropic = Anthropic
    install_module('anthropic', anth)

    monkeypatch.setenv('ANTHROPIC_API_KEY', 'test-key')

    router_mod = fresh_router_module()
    router = router_mod.ModelRouter()
    out = router.generate_text('hello', provider='anthropic', model='claude-3-5-sonnet', temperature=0.0, max_tokens=8)
    assert out == 'ok-claude'


def test_openai_generate_text_and_json(monkeypatch):
    clear_modules(['openai'])

    oai = types.ModuleType('openai')

    class RespText:
        def __init__(self, text):
            self.output_text = text

    class Responses:
        def create(self, **kwargs):
            # If JSON mode requested, return JSON string
            text_cfg = kwargs.get('text')
            if text_cfg is not None:
                return RespText(json.dumps({"ok": True}))
            return RespText('ok-openai')

    class OpenAI:
        def __init__(self):
            self.responses = Responses()

    # Expose OpenAI class at top level
    oai.OpenAI = OpenAI
    install_module('openai', oai)

    # Types for JSON mode imports
    json_obj_mod = types.ModuleType('openai.types.shared_params.response_format_json_object')
    class ResponseFormatJSONObject:
        def __init__(self, type=None):
            self.type = type
    json_obj_mod.ResponseFormatJSONObject = ResponseFormatJSONObject
    install_module('openai.types.shared_params.response_format_json_object', json_obj_mod)

    json_schema_mod = types.ModuleType('openai.types.responses.response_format_text_json_schema_config_param')
    class ResponseFormatTextJSONSchemaConfigParam:
        def __init__(self, type=None, name=None, schema=None, strict=None):
            self.type = type
            self.name = name
            self.schema = schema
            self.strict = strict
    json_schema_mod.ResponseFormatTextJSONSchemaConfigParam = ResponseFormatTextJSONSchemaConfigParam
    install_module('openai.types.responses.response_format_text_json_schema_config_param', json_schema_mod)

    monkeypatch.setenv('OPENAI_API_KEY', 'test-key')

    router_mod = fresh_router_module()
    router = router_mod.ModelRouter()

    # Text path
    t = router.generate_text('hello', provider='openai', model='gpt-4o-mini', temperature=0.0, max_tokens=8)
    assert t == 'ok-openai'

    # JSON path
    j = router.generate_json('{"ping":true}', provider='openai', model='gpt-4o-mini', schema={"title": "s"})
    assert j == {"ok": True}
# Ensure repository root is on sys.path so that `src` package is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
