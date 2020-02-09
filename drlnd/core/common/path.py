#!/usr/bin/env python3

from pathlib import Path


def get_project_root() -> Path:
    """
    Get project root directory with assumed structure as:
    ${PACKAGE_ROOT}/core/common/path.py
    """
    return Path(__file__).resolve().parent.parent.parent


def get_config_file() -> Path:
    """
    Get default config file.
    """
    return get_project_root()/'data/config/config.yaml'


def main():
    print(get_project_root())


if __name__ == '__main__':
    main()
