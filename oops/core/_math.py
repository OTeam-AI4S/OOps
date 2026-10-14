r"""Integer arithmetic for launch geometry."""


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _require_power_of_two(name: str, value: int) -> None:
    if value <= 0 or value & (value - 1):
        raise ValueError(f"{name} must be a positive power of two; got {value}")
