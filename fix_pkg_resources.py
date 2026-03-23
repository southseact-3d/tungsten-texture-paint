# Runtime hook to fix pkg_resources jaraco import error
# pkg_resources tries to import jaraco which may fail in frozen apps
import sys


# Pre-populate sys.modules with jaraco stubs to prevent import errors
class _JaracoModule:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


jaraco_stubs = [
    "jaraco",
    "jaraco.functools",
    "jaraco.text",
    "jaraco.context",
    "jaraco.path",
    "jaraco.envs",
]

for mod_name in jaraco_stubs:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = _JaracoModule()
