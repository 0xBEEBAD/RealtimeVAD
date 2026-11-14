"""RealtimeVAD: fast speech-end detection for low-latency voice assistants."""

from .iterator import VADIterator
from .vad import VAD, Block, Event, Quiet, QuietStream, Segment

__version__ = "0.1.1"

__all__ = ["VAD", "VADIterator", "Event", "Segment", "QuietStream", "Quiet", "Block", "__version__"]
