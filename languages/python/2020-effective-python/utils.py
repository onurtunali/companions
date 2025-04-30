from typing import Union


def to_str(arg: Union[str, bytes]) -> str:
    if isinstance(arg, str):
        return arg
    if isinstance(arg, bytes):
        return arg.decode("utf-8")
    raise TypeError("Argument must be str or bytes")
