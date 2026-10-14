"""Industry profiles (profiles.yaml): categories, complaints, features and sources per kind of product."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from config import PROJECT_ROOT

PROFILES_FILE = PROJECT_ROOT / "profiles.yaml"
DEFAULT_PROFILE = "consumer_product"
SOFTWARE = "software"


@dataclass(frozen=True)
class Profile:
    name: str
    version: int
    categories: dict[str, str]  # label -> zero-shot hypothesis
    praise: str
    discussion: str
    complaints: tuple[str, ...]
    complaints_need_non_positive: bool
    request: str | None
    features: dict[str, list[str]]
    sources: tuple[str, ...]

    def is_complaint(self, sentiment: str | None, category: str | None) -> bool:
        if sentiment == "negative":
            return True
        if category not in self.complaints:
            return False
        return not self.complaints_need_non_positive or sentiment != "positive"

    @property
    def version_tag(self) -> str:
        """Part of model_version. Empty for software, so existing software rows keep their version."""
        return "" if self.name == SOFTWARE else f".{self.name}{self.version}"


def _parse(name: str, raw: dict) -> Profile:
    categories = {str(k): str(v) for k, v in (raw.get("categories") or {}).items()}
    praise, discussion = raw.get("praise"), raw.get("discussion")
    complaints = tuple(raw.get("complaints") or [])
    request = raw.get("request")
    problems = [f"{label!r} is not one of its categories" for label in (praise, discussion, *complaints, request)
                if label is not None and label not in categories]
    if not categories or praise is None or discussion is None or not complaints:
        problems.append("needs categories, praise, discussion and complaints")
    if problems:
        raise ValueError(f"profiles.yaml: profile {name!r}: " + "; ".join(problems))
    features = {str(k): [str(s) for s in (v or [])] for k, v in (raw.get("features") or {}).items()}
    return Profile(
        name=name, version=int(raw.get("version", 1)), categories=categories, praise=praise, discussion=discussion,
        complaints=complaints, complaints_need_non_positive=bool(raw.get("complaints_need_non_positive", True)),
        request=request, features=features, sources=tuple(raw.get("sources") or ()),
    )


def parse_profile(name: str, raw: dict) -> Profile:
    """Validate one profile's raw YAML mapping (as in profiles.yaml); raises ValueError with the problems."""
    return _parse(name, raw)


def clear_cache() -> None:
    """Forget the loaded profiles (after a new one was added to profiles.yaml)."""
    _cached.cache_clear()


def load_profiles(path: Path = PROFILES_FILE) -> dict[str, Profile]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    profiles = {name: _parse(name, raw or {}) for name, raw in (data.get("profiles") or {}).items()}
    if DEFAULT_PROFILE not in profiles:
        raise ValueError(f"profiles.yaml: the fallback profile {DEFAULT_PROFILE!r} is required")
    return profiles


@lru_cache(maxsize=1)
def _cached() -> dict[str, Profile]:
    return load_profiles()


def get_profile(name: str | None) -> Profile:
    """The named profile, or the generic consumer_product fallback."""
    profiles = _cached()
    return profiles.get(name or DEFAULT_PROFILE, profiles[DEFAULT_PROFILE])


def profile_names() -> list[str]:
    return list(_cached())
