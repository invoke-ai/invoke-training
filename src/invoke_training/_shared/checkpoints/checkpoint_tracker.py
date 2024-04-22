import os
import shutil
import typing


class CheckpointTracker:
    """A utility class for managing checkpoint paths.

    Manages checkpoint paths of the following forms:
    - Checkpoint directories: `{base_dir}/{prefix}-epoch_{num_epochs}-step_{num_steps}`
    - Checkpoint files: `{base_dir}/{prefix}-epoch_{num_epochs}-step_{num_steps}{extension}`
    """

    def __init__(
        self,
        base_dir: str,
        prefix: str,
        extension: typing.Optional[str] = None,
        max_checkpoints: typing.Optional[int] = None,
        index_padding: int = 8,
    ):
        """Initialize a CheckpointTracker.

        Args:
            base_dir (str): The base checkpoint directory.
            prefix (str): A prefix applied to every checkpoint.
            extension (str, optional): If set, this is the file extension that will be applied to all checkpoints
                (usually one of ".pt", ".ckpt", or ".safetensors"). If None, then it will be assumed that we are
                managing checkpoint directories rather than files.
            max_checkpoints (typing.Optional[int], optional): The maximum number of checkpoints that should exist in
                base_dir.
            index_padding (int, optional): The length of the zero-padded epoch/step counts in the generated checkpoint
                names. E.g. index_padding=8 would produce checkpoint paths like
                "base_dir/prefix-epoch_00000001-step_00000001.ckpt".

        Raises:
            ValueError: If extension is provided, but it doesn not start with a '.'.
        """
        if extension is not None and not extension.startswith("."):
            raise ValueError(f"extension='{extension}' must start with a '.'.")

        self._base_dir = base_dir
        self._prefix = prefix
        self._extension = extension
        self._max_checkpoints = max_checkpoints
        self._index_padding = index_padding

    def prune(self, buffer_num: int = 1) -> int:
        """Delete checkpoint files and directories so that there are at most `max_checkpoints - buffer_num` checkpoints
        remaining. The checkpoints with the lowest step counts will be deleted.

        Args:
            buffer_num (int, optional): The number below `max_checkpoints` to 'free-up'.

        Returns:
            int: The number of checkpoints deleted.
        """
        if self._max_checkpoints is None:
            return 0

        checkpoints = os.listdir(self._base_dir)
        checkpoints = [p for p in checkpoints if p.startswith(self._prefix)]
        checkpoints = sorted(
            checkpoints,
            key=lambda x: int(os.path.splitext(x)[0].split("-step_")[-1]),
        )

        num_to_remove = len(checkpoints) - (self._max_checkpoints - buffer_num)
        if num_to_remove > 0:
            checkpoints_to_remove = checkpoints[:num_to_remove]

            for checkpoint_to_remove in checkpoints_to_remove:
                checkpoint_to_remove = os.path.join(self._base_dir, checkpoint_to_remove)
                if os.path.isfile(checkpoint_to_remove):
                    # Delete checkpoint file.
                    os.remove(checkpoint_to_remove)
                else:
                    # Delete checkpoint directory.
                    shutil.rmtree(checkpoint_to_remove)

        return max(0, num_to_remove)

    def get_path(self, epoch: int, step: int) -> str:
        """Get the checkpoint path for index `idx`.

        Args:
            epoch (int): The number of completed epochs.
            step (int): The number of completed training steps.

        Returns:
            str: The checkpoint path.
        """
        suffix = self._extension or ""
        return os.path.join(
            self._base_dir,
            f"{self._prefix.strip()}-epoch_{epoch:0>{self._index_padding}}-step_{step:0>{self._index_padding}}{suffix}",
        )
