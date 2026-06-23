import logging

from whisperjav.modules.scene_detection_backends.semantic_adapter import (
    SemanticClusteringAdapter,
)


def test_semantic_adapter_default_engine_logger_suppresses_info_prints(capsys):
    adapter = SemanticClusteringAdapter()

    adapter.logger.log(logging.INFO, "engine diagnostic")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
