class AspectRatioBucketManager:
    def __init__(self, target_resolution: tuple[int, int], start_dim: int, end_dim: int, divisible_by: int) -> None:
        self._buckets = self.build_aspect_ratio_buckets(
            target_resolution=target_resolution,
            start_dim=start_dim,
            end_dim=end_dim,
            divisible_by=divisible_by,
        )

    @classmethod
    def build_aspect_ratio_buckets(
        cls, target_resolution: tuple[int, int], start_dim: int, end_dim: int, divisible_by: int
    ) -> set[tuple[int, int]]:
        """Prepare a set of aspect ratios.

        Args:
            target_resolution (tuple[int, int]): The target resolution. All resolutions in the returned set will have <=
                the number of pixels in this target resolution.
            start_dim (int):
            end_dim (int):
            divisible_by (int): All dimensions in the returned set of resolutions will be divisible by `divisible_by`.

        Returns:
            set[tuple[int, int]]: The aspect ratio bucket resolutions.
        """
        # Validate target_resolution.
        target_res_h, target_res_w = target_resolution
        assert target_res_h % divisible_by == 0
        assert target_res_w % divisible_by == 0

        # Validate start_dim, end_dim.
        assert start_dim <= end_dim
        assert start_dim % divisible_by == 0
        assert end_dim % divisible_by == 0

        target_size = target_res_h * target_res_w

        buckets = set()

        height = start_dim
        while height <= end_dim:
            width = (target_size // height) // divisible_by * divisible_by
            buckets.add((height, width))
            buckets.add((width, height))

            height += divisible_by

        return buckets
