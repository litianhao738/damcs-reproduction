"""Run one project script with a named model profile from config/model_profiles.json.

Examples:
    python scripts/run_with_model.py --profile deepseek
    python scripts/run_with_model.py --profile kimi run.py --condition A --seed 4 --results_dir results/kimi_A4 --no_plot
Without a script, it prints the resolved configuration (no keys) and exits.
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from memory_system.model_profiles import ProfileError, profile_env  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--profile', required=True)
    parser.add_argument('script', nargs='?', default='scripts/check_llm_config.py')
    parser.add_argument('args', nargs=argparse.REMAINDER)
    options = parser.parse_args()
    try:
        env = profile_env(options.profile)
    except ProfileError as error:
        sys.exit(str(error))
    command = [sys.executable, '-B', options.script, *options.args]
    sys.exit(subprocess.run(command, cwd=ROOT, env={**os.environ, **env, 'PYTHONUNBUFFERED': '1'}).returncode)


if __name__ == '__main__':
    main()
