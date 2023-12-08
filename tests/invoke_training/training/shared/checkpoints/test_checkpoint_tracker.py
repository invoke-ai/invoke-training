import os
import tempfile
from pathlib import Path

import pytest

from invoke_training.training2.shared.checkpoints.checkpoint_tracker import CheckpointTracker


def test_checkpoint_tracker_get_path_file():
    """Test the CheckpointTracker.get_path(...) method with an extension."""
    checkpoint_tracker = CheckpointTracker(
        base_dir="base_dir",
        prefix="prefix",
        extension=".ckpt",
        index_padding=8,
    )

    path = checkpoint_tracker.get_path(55)

    assert Path(path) == Path("base_dir/prefix-00000055.ckpt")


def test_checkpoint_tracker_get_path_directory():
    """Test the CheckpointTracker.get_path(...) method without an extension."""
    checkpoint_tracker = CheckpointTracker(
        base_dir="base_dir",
        prefix="prefix",
        extension=None,
        index_padding=8,
    )

    path = checkpoint_tracker.get_path(55)

    assert Path(path) == Path("base_dir/prefix-00000055")


def test_checkpoint_tracker_bad_extension():
    """Test that CheckpointTracker raises a ValueError if an attempt is made to initialize it with an invalid
    extension.
    """
    with pytest.raises(ValueError):
        _ = CheckpointTracker(base_dir="base_dir", prefix="prefix", extension="ckpt")


def test_checkpoint_tracker_prune_files():
    """Test the CheckpointTracker.prune() method with checkpoint files."""
    with tempfile.TemporaryDirectory() as dir_name:
        checkpoint_tracker = CheckpointTracker(base_dir=dir_name, prefix="prefix", extension=".ckpt", max_checkpoints=5)
        # Create 6 checkpoints.
        for i in range(6):
            path = checkpoint_tracker.get_path(i)
            with open(path, "w") as f:
                f.write("hi")

        # Prune the 3 checkpoints with the lowest indices.
        num_pruned = checkpoint_tracker.prune(2)
        assert num_pruned == 3

        # Verify that the correct checkpoints were pruned.
        assert all([not os.path.exists(checkpoint_tracker.get_path(i)) for i in range(3)])
        assert all([os.path.exists(checkpoint_tracker.get_path(i)) for i in range(3, 6)])


def test_checkpoint_tracker_prune_directories():
    """Test the CheckpointTracker.prune() method with checkpoint directories."""
    with tempfile.TemporaryDirectory() as dir_name:
        checkpoint_tracker = CheckpointTracker(base_dir=dir_name, prefix="prefix", extension=None, max_checkpoints=5)
        # Create 6 checkpoints.
        for i in range(6):
            path = checkpoint_tracker.get_path(i)
            # Create checkpoint directory and add file to it.
            os.makedirs(path)
            with open(os.path.join(path, "tmp.txt"), "w") as f:
                f.write("hi")

        # Prune the 3 checkpoints with lowest indices.
        num_pruned = checkpoint_tracker.prune(2)
        assert num_pruned == 3

        # Verify that the correct checkpoints were pruned.
        assert all([not os.path.exists(checkpoint_tracker.get_path(i)) for i in range(3)])
        assert all([os.path.exists(checkpoint_tracker.get_path(i)) for i in range(3, 6)])


def test_checkpoint_tracker_prune_no_max():
    """Test that CheckpointTracker.prune() is a no-op when max_checkpoints is None."""
    with tempfile.TemporaryDirectory() as dir_name:
        checkpoint_tracker = CheckpointTracker(
            base_dir=dir_name, prefix="prefix", extension=".ckpt", max_checkpoints=None
        )
        # Create 6 checkpoints.
        for i in range(6):
            path = checkpoint_tracker.get_path(i)
            with open(path, "w") as f:
                f.write("hi")

        # Call prune, which should have no effect.
        num_pruned = checkpoint_tracker.prune(2)
        assert num_pruned == 0

        # Verify that no checkpoints were deleted.
        assert all([os.path.exists(checkpoint_tracker.get_path(i)) for i in range(6)])
