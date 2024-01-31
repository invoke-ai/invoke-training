# Dataset Architecture
Dataset handling is split into 3 layers of abstraction: Datasets, Transforms, and DataLoaders. Each is explained in more detail below.

## Datasets

Datasets implement the [torch.utils.data.Dataset](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files) interface.

Most dataset classes act as an abstraction over a specific dataset format.
 
## Transforms

Transforms are functions applied to data loaded by Datasets. For example, the `SDImageTransform` implements image augmentations for Stable Diffusion training.

Transforms are kept separate from the underlying datasets for several reasons:
- It is easier to write tests for isolated transforms.
- Modular transforms can often be re-used for multiple base datasets.
- Modular transforms make it easy to customize datasets for different situations. For example, you may want to wrap a dataset with one set of transforms initially to populate a cache, and then with a different set of transforms to read from the cache.

Transforms are applied to a dataset via the `TransformDataset` class.

## DataLoaders

The dataset classes (with composed transforms) are wrapped in a `torch.utils.data.DataLoader` that handles batch collation, multi-processing, etc.
