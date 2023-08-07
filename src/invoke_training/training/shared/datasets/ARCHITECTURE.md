# Dataset Architecture
Dataset handling is split into 3 layers of abstraction: `BaseImageCaptionReader`s, `ImageCaptionDataset`, and `DataLoader`. Each is explained in more detail below.

## `BaseImageCaptionReader`

`BaseImageCaptionReader` defines an interface that can be implemented by multiple sub-classes.

`BaseImageCaptionReader` sub-classes are intended as an abstraction over different dataset formats. They are responsible for loading image-caption pairs from disk. Readers implement the [torch Dataset interface](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files), i.e. they implement `__init__`, `__len__`, and `__getitem__`.

Two examples of concrete reader implementations are the `HFHubImageCaptionReader` and the `HFDirImageCaptionReader`.

## `ImageCaptionDataset`

An `ImageCaptionDataset` is a wrapper for a `BaseImageCaptionReader` object. The `ImageCaptionDataset` is responsible for example-level operations that are agnostic to the underlying dataset format. For example, image augmentations are handled at this layer.

## `DataLoader`

The `ImageCaptionDataset` is wrapped in a `torch.utils.data.DataLoader` that handles batch collation, multi-processing, etc.
