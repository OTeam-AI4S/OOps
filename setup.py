from pathlib import Path
from runpy import run_path

from setuptools import setup

run_path(str(Path(__file__).parent / "oops" / "_edition.py"))["_warn_internal_users"]()
setup()
