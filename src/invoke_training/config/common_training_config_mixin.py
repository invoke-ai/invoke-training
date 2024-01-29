from pydantic import BaseModel


class CommonTrainingConfigMixin(BaseModel):
    """A mixin class that contains configs that are likely to be shared by many training pipelines."""

    max_train_steps: int | None = None
    """Total number of training steps to perform. One training step is one gradient update.

    One of `max_train_steps` or `max_train_epochs` should be set.
    """

    max_train_epochs: int | None = None
    """Total number of training epochs to perform. One epoch is one pass over the entire dataset.

    One of `max_train_steps` or `max_train_epochs` should be set.
    """

    save_every_n_epochs: int | None = None
    """The interval (in epochs) at which to save checkpoints.

    One of `save_every_n_epochs` or `save_every_n_steps` should be set.
    """

    save_every_n_steps: int | None = None
    """The interval (in steps) at which to save checkpoints.

    One of `save_every_n_epochs` or `save_every_n_steps` should be set.
    """

    validate_every_n_epochs: int | None = None
    """The interval (in epochs) at which validation images will be generated.

    One of `validate_every_n_epochs` or `validate_every_n_steps` should be set.
    """

    validate_every_n_steps: int | None = None
    """The interval (in steps) at which validation images will be generated.

    One of `validate_every_n_epochs` or `validate_every_n_steps` should be set.
    """
