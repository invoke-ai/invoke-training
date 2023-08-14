TODO(ryand): Update this document. It is out-of-date.

# Dataset Architecture
Dataset handling is split into 3 layers of abstraction: Readers, Datasets, and DataLoaders. Each is explained in more detail below.

## Readers

`BaseImageCaptionReader` defines a reader interface that can be implemented by multiple sub-classes.

`BaseImageCaptionReader` sub-classes are intended as an abstraction over different dataset formats. They are responsible for loading image-caption pairs from disk. Readers implement the [torch Dataset interface](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files), i.e. they implement `__init__`, `__len__`, and `__getitem__`.

Two examples of concrete reader implementations are the `HFHubImageCaptionReader` and the `HFDirImageCaptionReader`.

## Datasets

Dataset classes wrap reader classes and are responsible for example-level operations that are agnostic to the underlying dataset format.

As an example `ImageCaptionSDDataset` is a wrapper for a `BaseImageCaptionReader` object that implements image augmentations and caption tokenization.

## DataLoaders

The dataset classes are wrapped in a `torch.utils.data.DataLoader` that handles batch collation, multi-processing, etc.
