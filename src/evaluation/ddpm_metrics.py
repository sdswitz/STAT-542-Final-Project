from __future__ import annotations

from typing import Any


def compute_kid_for_run(*args, **kwargs) -> dict[str, Any]:
    """Compute KID for a saved run.

    This is intentionally left blank because the project needs to decide:
    - which real-image reference split to use
    - how many generated samples to evaluate
    - which feature extractor/package to standardize on
    - how to store confidence intervals
    """
    raise NotImplementedError("Choose and implement the project-wide KID protocol here.")


def compute_fid_for_run(*args, **kwargs) -> dict[str, Any]:
    """Compute FID for a saved run after the KID protocol is settled."""
    raise NotImplementedError("Choose and implement the project-wide FID protocol here.")
