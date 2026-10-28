"""Names used by historical local PI05 checkpoints; no runtime model changes."""

import re
from collections.abc import Iterable

_JOINT_KEY = re.compile(r"^(.*paligemma_with_expert\.)joint_layers\.(\d+)\.(vlm|expert)\.(.+)$")
_VISION_KEY = re.compile(r"^(.*paligemma_with_expert\.paligemma\.model\.vision_tower\.)vision_model\.(.+)$")


def canonical_pi05_key(name: str) -> str:
    """Translate legacy joint decoder ownership and the old SigLIP container."""
    if match := _JOINT_KEY.match(name):
        prefix, index, branch, suffix = match.groups()
        owner = "paligemma.model.language_model" if branch == "vlm" else "gemma_expert.model"
        name = f"{prefix}{owner}.layers.{index}.{suffix}"
    if match := _VISION_KEY.match(name):
        name = "".join(match.groups())
    return name


def pi05_checkpoint_key_map(names: Iterable[str]) -> dict[str, str]:
    """Map canonical names to saved names, rejecting ambiguous checkpoint aliases."""
    result = {}
    for name in names:
        canonical = canonical_pi05_key(name)
        if canonical in result and result[canonical] != name:
            raise ValueError(f"Duplicate PI05 checkpoint key after conversion: {canonical}")
        result[canonical] = name
    return result


def canonical_pi05_state_dict[T](state_dict: dict[str, T]) -> dict[str, T]:
    """Rename checkpoint tensors without copying or dropping any entry."""
    return {name: state_dict[saved] for name, saved in pi05_checkpoint_key_map(state_dict).items()}
