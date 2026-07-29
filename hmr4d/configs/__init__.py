from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from hydra_zen import builds

import argparse
from hydra import compose, initialize_config_module
import os

os.environ["HYDRA_FULL_ERROR"] = "1"

MainStore = ConfigStore.instance()


def register_store_footmr():
    """group 옵션을 MainStore에 등록합니다."""
    from . import store_footmr


def parse_args_to_cfg():
    """
    최소한의 Hydra API로 인자를 해석하고 cfg를 반환합니다.
    log 파일 계층을 만드는 ``_run_hydra``는 실행하지 않습니다.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", "-cn", default="train")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Any key=value arguments to override config values (use dots for.nested=overrides)",
    )
    args = parser.parse_args()

    # 설정 구성
    with initialize_config_module(version_base="1.3", config_module=f"hmr4d.configs"):
        cfg = compose(config_name=args.config_name, overrides=args.overrides)

    return cfg
