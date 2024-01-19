from invoke_training.training._shared.data.utils.resolution import Resolution


class AspectRatioBucketManager:
    def __init__(self, buckets: set[Resolution]):
        self.buckets = buckets

    @classmethod
    def from_constraints(
        cls, target_resolution: int, start_dim: int, end_dim: int, divisible_by: int
    ) -> "AspectRatioBucketManager":
        buckets = cls.build_aspect_ratio_buckets(
            target_resolution=target_resolution,
            start_dim=start_dim,
            end_dim=end_dim,
            divisible_by=divisible_by,
        )
        return cls(buckets)

    @classmethod
    def build_aspect_ratio_buckets(
        cls, target_resolution: int, start_dim: int, end_dim: int, divisible_by: int
    ) -> set[Resolution]:
        """Prepare a set of aspect ratios.

        Args:
            target_resolution (Resolution): All resolutions in the returned set will aim to have close to
                (but <=) `target_resolution * target_resolution` pixels.
            start_dim (int):
            end_dim (int):
            divisible_by (int): All dimensions in the returned set of resolutions will be divisible by `divisible_by`.

        Returns:
            set[tuple[int, int]]: The aspect ratio bucket resolutions.
        """
        # Validate target_resolution.
        assert target_resolution % divisible_by == 0

        # Validate start_dim, end_dim.
        assert start_dim <= end_dim
        assert start_dim % divisible_by == 0
        assert end_dim % divisible_by == 0

        target_size = target_resolution * target_resolution

        buckets = set()

        height = start_dim
        while height <= end_dim:
            width = (target_size // height) // divisible_by * divisible_by
            buckets.add(Resolution(height, width))
            buckets.add(Resolution(width, height))

            height += divisible_by

        return buckets

    def get_aspect_ratio_bucket(self, resolution: Resolution):
        """Get the bucket with the closest aspect ratio to 'resolution'."""
        # Note: If this is ever found to be a bottleneck, there is a clearly-more-efficient implementation using bisect.
        return min(self.buckets, key=lambda x: abs(x.aspect_ratio() - resolution.aspect_ratio()))
