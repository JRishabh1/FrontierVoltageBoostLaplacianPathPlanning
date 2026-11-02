"""Build the optional CPU A* extension: python benchmarks/build_astar.py."""
import os
from pathlib import Path
import shlex
import subprocess
import sys
import sysconfig
import tempfile


def main():
    import pybind11
    root = Path(__file__).resolve().parents[1]
    destination = root / 'Algorithms' / ('_astar_native' + sysconfig.get_config_var('EXT_SUFFIX'))
    compiler = shlex.split(os.environ.get('CXX', 'c++'))
    with tempfile.TemporaryDirectory(prefix='astar-build-') as folder:
        output = Path(folder) / destination.name
        command = compiler + ['-O3', '-std=c++17', '-shared', '-fPIC',
            '-I' + pybind11.get_include(), '-I' + sysconfig.get_path('include'),
            str(root / 'Algorithms/native/astar.cpp'), '-o', str(output)]
        if sys.platform == 'darwin':
            command += ['-undefined', 'dynamic_lookup']
        subprocess.run(command, check=True)
        destination.write_bytes(output.read_bytes())
    print(f'Built {destination}; restart the app to use cpp-astar.')


if __name__ == '__main__':
    main()
