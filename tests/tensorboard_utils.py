"""Keep real TensorBoard writers and their files inside temporary test folders."""

from pathlib import Path
import tempfile
from unittest.mock import patch

from tensorboardX import SummaryWriter


def isolate_tensorboard_writers(test_class):
    directory = tempfile.TemporaryDirectory(prefix="scorch-tensorboard-")
    root = Path(directory.name)
    writers = []
    test_class.addClassCleanup(directory.cleanup)

    def create_writer(logdir=None, **kwargs):
        requested = kwargs.pop("log_dir", logdir)
        # Preserve the requested folder name so invalid run names still fail.
        destination = root / Path(requested).name if requested is not None else root / "default"
        writer = SummaryWriter(log_dir=str(destination), **kwargs)
        writers.append(writer)
        return writer

    patcher = patch("Trainer.scorch_trainer.SummaryWriter", side_effect=create_writer)
    patcher.start()
    test_class.addClassCleanup(patcher.stop)

    def close_writers():
        for writer in writers:
            writer.close()

    # LIFO cleanup closes files before deleting their directories on Windows.
    test_class.addClassCleanup(close_writers)
    return root
