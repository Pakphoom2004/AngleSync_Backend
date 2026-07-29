from typing import Any, Optional

_keypoints_cache: dict = {}


def store_keypoints(filename: str, keypoints: Any) -> None:
    if not filename:
        return
    _keypoints_cache[filename] = keypoints


def get_keypoints(filename: str) -> Optional[Any]:
    if not filename:
        return None
    return _keypoints_cache.get(filename)