"""Keep real TensorBoard loggers and their files inside temporary test folders."""

from pathlib import Path
import tempfile
from unittest.mock import patch

from Trainer.tensorboard_logger import TensorBoardLogger


def isolate_tensorboard_writers(test_class):
    directory = tempfile.TemporaryDirectory(prefix="scorch-tensorboard-")
    root = Path(directory.name)
    loggers = []
    test_class.addClassCleanup(directory.cleanup)

    def create_logger(log_dir=None, *, run_name=None):
        # Keep the real timestamp/naming behavior while redirecting defaults.
        logger = TensorBoardLogger(log_dir=log_dir, run_name=run_name, base_dir=root)
        loggers.append(logger)
        return logger

    patcher = patch("Trainer.scorch_trainer.TensorBoardLogger", side_effect=create_logger)
    patcher.start()
    test_class.addClassCleanup(patcher.stop)

    def close_loggers():
        for logger in loggers:
            logger.close()

    # LIFO cleanup closes files before deleting their directories on Windows.
    test_class.addClassCleanup(close_loggers)
    return root
