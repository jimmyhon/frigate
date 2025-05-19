import logging

from typing_extensions import Literal

from frigate.detectors.detection_api import DetectionApi
from frigate.detectors.detector_config import BaseDetectorConfig
from .tflite import load_delegate_interpreter, finish_init, tflite_detect_raw


logger = logging.getLogger(__name__)

# Use _tfl suffix to default tflite model
DETECTOR_KEY = "teflon_tfl"


class TeflonDetectorConfig(BaseDetectorConfig):
    type: Literal[DETECTOR_KEY]


class TeflonTfl(DetectionApi):
    type_key = DETECTOR_KEY

    def __init__(self, detector_config: TeflonDetectorConfig):
        # Location in Debian's mesa-teflon-delegate
        delegate_library = "/usr/lib/teflon/libteflon.so"
        device_config = {}

        interpreter = load_delegate_interpreter(delegate_library, detector_config, device_config)
        finish_init(self, interpreter)

    def detect_raw(self, tensor_input):
        return tflite_detect_raw(self, tensor_input)

