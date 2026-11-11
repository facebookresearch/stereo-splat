# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import re
from pathlib import Path

from stereosplat.datasets.factory import load_split_file

REPO = Path(__file__).resolve().parents[1]


def test_pyproject_lists_every_package() -> None:
    text = (REPO / "pyproject.toml").read_text()
    block = re.search(r"^packages = \[(.*?)\]", text, re.S | re.M)
    assert block is not None
    listed = set(re.findall(r'"([\w.]+)"', block.group(1)))
    found = {
        ".".join(p.parent.relative_to(REPO).parts)
        for p in (REPO / "stereosplat").rglob("__init__.py")
    }
    assert found | {"stereosplat.splits"} == listed
    assert '"stereosplat.splits" = "data/splits"' in text


def test_shipped_splits_resolve() -> None:
    for path in sorted((REPO / "data" / "splits").glob("*/*.txt")):
        rel = path.relative_to(REPO).as_posix()
        assert load_split_file(rel), rel
