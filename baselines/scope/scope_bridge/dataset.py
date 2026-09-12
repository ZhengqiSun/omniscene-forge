from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from .schema import ScopeSampleV1, load_manifest, resolve_path


class ScopeManifestDataset:
    """A deliberately narrow row projector; inference cannot even expose GT paths."""

    def __init__(self, manifest: str | Path, mode: str, splits: set[str] | None = None):
        if mode not in {"train", "infer"}:
            raise ValueError(mode)
        self.manifest = Path(manifest).resolve()
        self.mode = mode
        self.samples = [s for s in load_manifest(manifest) if not splits or s.split in splits]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        row = sample.model_projection(self.mode)
        for key in ("initial_image", "raw_video", "raw_action_path", "scope_action_path", "target_video", "target_latent"):
            if key in row:
                row[key] = str(resolve_path(self.manifest, row[key]))
        return row

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for i in range(len(self)):
            yield self[i]


def assert_train_split(samples: list[ScopeSampleV1], debug_allow_test: bool = False) -> bool:
    has_test = any(sample.split == "test" for sample in samples)
    if has_test and not debug_allow_test:
        raise ValueError("refusing to train on split=test; use --debug-allow-test only for explicit debug")
    return has_test
