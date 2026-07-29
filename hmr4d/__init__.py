import os
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[1]


def os_chdir_to_proj_root():
    """서로 다른 디렉터리에서 notebook을 실행할 때 사용한다."""
    os.chdir(PROJ_ROOT)
