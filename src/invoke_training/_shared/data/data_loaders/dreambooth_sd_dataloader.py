import typing

from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler

from invoke_training._shared.data.data_loaders.image_caption_sd_dataloader import (
    build_aspect_ratio_bucket_manager,
    sd_image_caption_collate_fn,
)
from invoke_training._shared.data.datasets.image_dir_dataset import ImageDirDataset
from invoke_training._shared.data.datasets.transform_dataset import TransformDataset
from invoke_training._shared.data.samplers.aspect_ratio_bucket_batch_sampler import AspectRatioBucketBatchSampler
from invoke_training._shared.data.samplers.batch_offset_sampler import BatchOffsetSampler
from invoke_training._shared.data.samplers.concat_sampler import ConcatSampler
from invoke_training._shared.data.samplers.interleaved_sampler import InterleavedSampler
from invoke_training._shared.data.samplers.offset_sampler import OffsetSampler
from invoke_training._shared.data.transforms.constant_field_transform import ConstantFieldTransform
from invoke_training._shared.data.transforms.drop_field_transform import DropFieldTransform
from invoke_training._shared.data.transforms.load_cache_transform import LoadCacheTransform
from invoke_training._shared.data.transforms.sd_image_transform import SDImageTransform
from invoke_training._shared.data.transforms.tensor_disk_cache import TensorDiskCache
from invoke_training.config.data.data_loader_config import DreamboothSDDataLoaderConfig


def build_dreambooth_sd_dataloader(
    config: DreamboothSDDataLoaderConfig,
    batch_size: int,
    text_encoder_output_cache_dir: typing.Optional[str] = None,
    text_encoder_cache_field_to_output_field: typing.Optional[dict[str, str]] = None,
    vae_output_cache_dir: typing.Optional[str] = None,
    shuffle: bool = True,
    sequential_batching: bool = False,
) -> DataLoader:
    """Construct a DataLoader for a DreamBooth dataset for Stable Diffusion XL.

    Args:
        config (DreamboothSDDataLoaderConfig):
        batch_size (int):
        text_encoder_output_cache_dir (str, optional): The directory where text encoder outputs are cached and should be
            loaded from.
        vae_output_cache_dir (str, optional): The directory where VAE outputs are cached and should be loaded from. If
            set, then the image augmentation transforms will be skipped, and the image will not be copied to VRAM.
        shuffle (bool, optional): Whether to shuffle the dataset order.
        sequential_batching (bool, optional): If True, the internal dataset will be processed sequentially rather than
            interleaving class and instance examples. This is intended to be used when processing the entire dataset for
            caching purposes. Defaults to False.

    Returns:
        DataLoader
    """
    # Prepare instance dataset.
    base_instance_dataset = ImageDirDataset(
        config.instance_dataset.dataset_dir,
        id_prefix="instance_",
        keep_in_memory=config.instance_dataset.keep_in_memory,
    )
    instance_dataset = TransformDataset(
        base_instance_dataset,
        [
            ConstantFieldTransform("caption", config.instance_caption),
            ConstantFieldTransform("loss_weight", 1.0),
        ],
    )
    datasets = [instance_dataset]

    # Prepare class dataset.
    base_class_dataset = None
    class_dataset = None
    if config.class_dataset is not None:
        base_class_dataset = ImageDirDataset(
            config.class_dataset.dataset_dir, id_prefix="class_", keep_in_memory=config.class_dataset.keep_in_memory
        )
        class_dataset = TransformDataset(
            base_class_dataset,
            [
                ConstantFieldTransform("caption", config.class_caption),
                ConstantFieldTransform("loss_weight", config.class_data_loss_weight),
            ],
        )
        datasets.append(class_dataset)

    # Merge instance dataset and class dataset.
    merged_dataset = ConcatDataset(datasets)

    # Initialize either the fixed target resolution or aspect ratio buckets.
    target_resolution = None
    aspect_ratio_bucket_manager = None
    instance_sampler = None
    class_sampler = None
    if config.aspect_ratio_buckets is None:
        target_resolution = config.resolution
        # TODO(ryand): Provide a seeded generator.
        instance_sampler = RandomSampler(instance_dataset) if shuffle else SequentialSampler(instance_dataset)
        if base_class_dataset is not None:
            class_sampler = RandomSampler(class_dataset) if shuffle else SequentialSampler(class_dataset)
            class_sampler = OffsetSampler(class_sampler, offset=len(base_instance_dataset))
    else:
        aspect_ratio_bucket_manager = build_aspect_ratio_bucket_manager(config=config.aspect_ratio_buckets)
        # TODO(ryand): Drill-down the seed parameter rather than hard-coding to 0 here.
        instance_sampler = AspectRatioBucketBatchSampler.from_image_sizes(
            bucket_manager=aspect_ratio_bucket_manager,
            image_sizes=base_instance_dataset.get_image_dimensions(),
            batch_size=batch_size,
            shuffle=shuffle,
            seed=0,
        )
        if base_class_dataset is not None:
            class_sampler = AspectRatioBucketBatchSampler.from_image_sizes(
                bucket_manager=aspect_ratio_bucket_manager,
                image_sizes=base_class_dataset.get_image_dimensions(),
                batch_size=batch_size,
                shuffle=shuffle,
                seed=0,
            )
            class_sampler = BatchOffsetSampler(class_sampler, offset=len(base_instance_dataset))

    # Add transforms to the merged dataset.
    all_transforms = []
    if vae_output_cache_dir is None:
        all_transforms.append(
            SDImageTransform(
                image_field_names=["image"],
                fields_to_normalize_to_range_minus_one_to_one=["image"],
                resolution=target_resolution,
                aspect_ratio_bucket_manager=aspect_ratio_bucket_manager,
                center_crop=config.center_crop,
                random_flip=config.random_flip,
            )
        )
    else:
        vae_cache = TensorDiskCache(vae_output_cache_dir)
        all_transforms.append(
            LoadCacheTransform(
                cache=vae_cache,
                cache_key_field="id",
                cache_field_to_output_field={
                    "vae_output": "vae_output",
                    "original_size_hw": "original_size_hw",
                    "crop_top_left_yx": "crop_top_left_yx",
                },
            )
        )
        # We drop the image to avoid having to either convert from PIL, or handle PIL batch collation.
        all_transforms.append(DropFieldTransform("image"))

    if text_encoder_output_cache_dir is not None:
        assert text_encoder_cache_field_to_output_field is not None
        text_encoder_cache = TensorDiskCache(text_encoder_output_cache_dir)
        all_transforms.append(
            LoadCacheTransform(
                cache=text_encoder_cache,
                cache_key_field="id",
                cache_field_to_output_field=text_encoder_cache_field_to_output_field,
            )
        )

    merged_dataset = TransformDataset(merged_dataset, all_transforms)

    # Choose between sequential vs. interleaved merging of the instance and class samplers.
    # Sequential sampling is typically used to populate a cache, because it guarantees that all examples will be
    # included in an epoch.
    samplers = [instance_sampler]
    if class_sampler is not None:
        samplers.append(class_sampler)
    if sequential_batching:
        sampler = ConcatSampler(samplers)
    else:
        sampler = InterleavedSampler(samplers)

    if config.aspect_ratio_buckets is None:
        return DataLoader(
            merged_dataset,
            sampler=sampler,
            collate_fn=sd_image_caption_collate_fn,
            batch_size=batch_size,
            num_workers=config.dataloader_num_workers,
        )
    else:
        # If config.aspect_ratio_buckets is not None, then we are using a batch sampler.
        return DataLoader(
            merged_dataset,
            batch_sampler=sampler,
            collate_fn=sd_image_caption_collate_fn,
            num_workers=config.dataloader_num_workers,
        )
