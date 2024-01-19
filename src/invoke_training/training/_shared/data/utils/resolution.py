from typing import Union


class Resolution:
    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width

    @classmethod
    def parse(cls, resolution: Union[int, tuple[int, int], "Resolution"]):
        """Initialize a Resolution object from another type."""
        if isinstance(resolution, int):
            # Assume square resolution.
            return cls(resolution, resolution)
        elif isinstance(resolution, tuple):
            height, width = resolution
            return cls(height, width)
        elif isinstance(resolution, cls):
            return cls(resolution.height, resolution.width)
        else:
            raise ValueError(f"Unsupported resolution type: '{type(resolution)}'.")

    def aspect_ratio(self) -> float:
        return self.height / self.width

    def to_tuple(self) -> tuple[int, int]:
        return (self.height, self.width)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Resolution):
            raise NotImplementedError(f"Not implemented for other type: {type(other)}.")
        return self.to_tuple() == other.to_tuple()

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Resolution):
            raise NotImplementedError(f"Not implemented for other type: {type(other)}.")
        return self.to_tuple() < other.to_tuple()

    def __hash__(self) -> int:
        return hash(self.to_tuple())
