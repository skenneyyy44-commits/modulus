# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint metadata must survive missing distribution metadata."""

import json
import tarfile
import zipfile
from importlib import metadata
from types import SimpleNamespace

import pytest
import torch

from physicsnemo import Module
from physicsnemo.core.version_check import get_installed_version


class CheckpointModel(Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([3.0]))


@pytest.mark.parametrize("legacy_format", [False, True])
@pytest.mark.parametrize("lookup", ["exact", "prefix", "missing"])
def test_checkpoint_metadata_roundtrip(tmp_path, monkeypatch, lookup, legacy_format):
    """Save real archives and restore weights for all version lookup paths."""
    get_installed_version.cache_clear()

    def version(name):
        if name == "nvidia-physicsnemo" and lookup != "exact":
            raise metadata.PackageNotFoundError(name)
        return "1.4.0"

    distributions = (
        [
            SimpleNamespace(
                metadata={"Name": "nvidia-physicsnemo-dev"}, version="1.4.0.dev1"
            )
        ]
        if lookup == "prefix"
        else []
    )
    monkeypatch.setattr(metadata, "version", version)
    monkeypatch.setattr(metadata, "distributions", lambda: distributions)
    expected = {"exact": "1.4.0", "prefix": "1.4.0.dev1", "missing": "unknown"}[lookup]
    try:
        path = str(tmp_path / "checkpoint.mdlus")
        CheckpointModel().save(path, legacy_format=legacy_format)
        if legacy_format:
            with tarfile.open(path) as archive:
                saved = json.load(archive.extractfile("metadata.json"))
        else:
            with zipfile.ZipFile(path) as archive:
                saved = json.loads(archive.read("metadata.json"))
        assert saved["physicsnemo_version"] == expected
        restored = CheckpointModel()
        with torch.no_grad():
            restored.weight.zero_()
        restored.load(path)
        torch.testing.assert_close(restored.weight, torch.tensor([3.0]))
    finally:
        get_installed_version.cache_clear()
