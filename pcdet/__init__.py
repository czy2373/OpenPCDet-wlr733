import subprocess
from pathlib import Path

from .version import __version__

__all__ = [
    '__version__'
]


def get_git_commit_number():
    repo_root = (Path(__file__).resolve().parent / '..').resolve()
    if not (repo_root / '.git').exists():
        return '0000000'

    cmd_out = subprocess.run(
        ['git', 'rev-parse', 'HEAD'],
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if cmd_out.returncode != 0:
        return '0000000'
    git_commit_number = cmd_out.stdout.decode('utf-8')[:7]
    return git_commit_number


script_version = get_git_commit_number()


if script_version not in __version__:
    __version__ = __version__ + '+py%s' % script_version
