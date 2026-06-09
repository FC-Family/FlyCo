from .prompt_session import PromptSession
from .depth_guidance import DepthGuidancePolicy

__all__ = [
    "PromptSession",
    "DepthGuidancePolicy",
    "TrackingPipeline",
]


def __getattr__(name):
    if name == "TrackingPipeline":
        from .tracking_pipeline import TrackingPipeline

        return TrackingPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
