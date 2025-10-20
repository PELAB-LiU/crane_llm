from collections.abc import Mapping, Set, Sequence
import json
from pathlib import Path
from typing import Tuple

# TODO: I don't think it's necessary to have a nested dictionary for this. It's overly complicated now.
# i.e. `{"sklearn": {"sklearn.base.clone": "sklearn.clone"}}` could just be `{"sklearn.base.clone": "sklearn.clone"}`
# as the overarching key is present in that one too, so there won't be any key clashes unless we support two packages
# with the same name.

DEFAULT_ALIAS_MAPPING_PATH = Path(__file__).parent.resolve()\
    .joinpath('resources')\
    .joinpath('internal_to_public_alias_mapping.json')

_internal_to_public_alias_mapping: Mapping | None = None
_public_aliases: Set | None = None


def reload_internal_to_public_alias_mapping(alias_mapping_path: Path = DEFAULT_ALIAS_MAPPING_PATH):
    """Reloads the global mapping with the provided path."""
    global _internal_to_public_alias_mapping, _public_aliases
    with open(alias_mapping_path, 'r') as alias_mapping_file:
        j_data = alias_mapping_file.read()
        _internal_to_public_alias_mapping = json.loads(j_data)
        _public_aliases = {key: set(value.values()) for key, value
                           in _internal_to_public_alias_mapping.items()}


def get_internal_to_public_alias_mapping():
    """
    Loads the global mapping if it wasn't yet,
    and returns the loaded mapping.
    """
    global _internal_to_public_alias_mapping, _public_aliases
    if _internal_to_public_alias_mapping is None or _public_aliases is None:
        reload_internal_to_public_alias_mapping()
    return _internal_to_public_alias_mapping, _public_aliases


def try_map_to_public_alias(internal_alias: str) -> Tuple[bool, str]:
    """
    Maps the provided internal alias to the public one if
    the mapping is defined, otherwise it returns the internal alias.
    The boolean specifies whether the mapping succeeded.
    """
    _mapping, _ = get_internal_to_public_alias_mapping()
    module = internal_alias.split(".")[0]
    if module in _mapping and internal_alias in _mapping[module]:
        public_alias = _mapping[module][internal_alias]
        return True, public_alias
    return False, internal_alias


def is_public_alias(alias: str) -> bool:
    root = alias.split(".")[0]
    _, public_aliases = get_internal_to_public_alias_mapping()
    if root not in public_aliases:
        return False
    public_aliases = public_aliases[root]
    is_public = alias in public_aliases
    return is_public


def map_to_public_alias(
    internal_alias: str,
    raise_on_missing_alias: bool = False,
    ignore_when_already_public_alias: bool = True
) -> str:
    """
    Maps the provided internal alias onto the public one, if there
    exists one, otherwise it returns the internal alias.
    """
    if ignore_when_already_public_alias and is_public_alias(internal_alias):
        return internal_alias
    success, public_alias = try_map_to_public_alias(internal_alias)
    if raise_on_missing_alias and not success:
        raise KeyError(internal_alias)
    return public_alias
 