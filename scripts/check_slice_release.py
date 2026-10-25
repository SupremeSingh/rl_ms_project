"""Build/install a wheel outside the checkout and run its CPU acceptance suite.

Uses the caller's existing dependencies; no dependency downloads or environment
mutation. CI supplies a fresh environment. Never publishes a package.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def check(output):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = dict(passed=False, scope='Installed-wheel CPU acceptance; no GPU, model download or publication')
    log_path = output / 'checks.log'
    try:
        with tempfile.TemporaryDirectory(prefix='slice-release-') as temporary, log_path.open('w') as log:
            work = Path(temporary)
            target = work / 'installed'
            tests = work / 'acceptance'
            tests.mkdir()
            env = dict(os.environ, PIP_NO_INDEX='1', PIP_DISABLE_PIP_VERSION_CHECK='1')
            env.pop('PYTHONPATH', None)

            def run(argv, cwd=work):
                log.write('$ ' + ' '.join(map(str, argv)) + '\n')
                log.flush()
                subprocess.run(list(map(str, argv)), cwd=cwd, env=env,
                               stdout=log, stderr=subprocess.STDOUT, check=True)

            run([sys.executable, '-m', 'pip', 'wheel', '--no-deps', '--no-build-isolation',
                 ROOT / 'packages/slice-critic', '--wheel-dir', output])
            wheel, = output.glob('slice_critic-*.whl')
            run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--no-compile', '--target', target, wheel])
            env['PYTHONPATH'] = str(target)
            probe = '''import importlib.abc, importlib.metadata as m, json, pathlib, sys
class BlockProject(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'math_rl', 'verl', 'jlens'}:
            raise ImportError('Standalone package must not import project: ' + fullname)
sys.meta_path.insert(0, BlockProject())
import slice_critic
assert pathlib.Path(slice_critic.__file__).is_relative_to(pathlib.Path(sys.argv[1]))
for name in slice_critic.__all__: getattr(slice_critic, name)
assert slice_critic.__version__ == m.version('slice-critic')
print(json.dumps({'version': slice_critic.__version__, 'python': sys.version,
                 'dependencies': {n: m.version(n) for n in ('torch', 'numpy', 'transformers', 'datasets')}}))
'''
            probe_result = subprocess.run([sys.executable, '-c', probe, str(target)], cwd=work,
                                          env=env, text=True, capture_output=True, check=True)
            result['runtime'] = json.loads(probe_result.stdout)
            # Copy tests, not repository import paths. Source copy is only for AST boundary checks.
            shutil.copytree(ROOT / 'packages/slice-critic/tests', tests / 'tests',
                            ignore=shutil.ignore_patterns('__pycache__'))
            # Enforce independence throughout training/scoring tests, not just imports.
            (tests / 'tests/conftest.py').write_text(probe.split('import slice_critic')[0])
            shutil.copytree(target / 'slice_critic', tests / 'src/slice_critic')
            (tests / 'pytest.ini').write_text('[pytest]\ntestpaths = tests\n')
            run([sys.executable, '-m', 'pytest', '-q', 'tests'], cwd=tests)
            config = work / 'preview.json'
            config.write_text(json.dumps(dict(dataset='not-downloaded', max_questions=40)))
            run([sys.executable, '-m', 'slice_critic', 'preview', '--config', config])
            result.update(passed=True, wheel=wheel.name,
                          wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest())
    except BaseException as exc:
        result['error'] = str(exc)
        if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
            with log_path.open('a') as log:
                log.write(str(exc.stderr))
        raise
    finally:
        (output / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True, help='New directory for wheel, report and logs')
    check(parser.parse_args().out)
