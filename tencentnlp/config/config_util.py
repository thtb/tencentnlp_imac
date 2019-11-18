NAME_PREFIX = "__"


def to_storage_name(real_name: str) -> str:
    return NAME_PREFIX + real_name


def to_real_name(storage_name: str) -> str:
    if storage_name.startswith(NAME_PREFIX):
        return storage_name[len(NAME_PREFIX):]
    return storage_name
