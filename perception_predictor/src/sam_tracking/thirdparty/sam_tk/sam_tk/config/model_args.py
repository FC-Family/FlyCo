import importlib.util
import os
from pathlib import Path


_THIS_FILE = Path(__file__).resolve()
_SAM_TK_ROOT = _THIS_FILE.parents[1]
_PACKAGE_ROOT = _THIS_FILE.parents[4]
_VENDORED_SAM2_ROOT = _PACKAGE_ROOT / "thirdparty" / "sam2-realtime"
_CKPT_ROOT = _PACKAGE_ROOT / "ckpt"
def _as_str(path: Path) -> str:
    return str(path)


def _installed_sam2_package_root():
    spec = importlib.util.find_spec("sam2")
    if spec is None or not spec.origin:
        return None
    return Path(spec.origin).resolve().parent


def _require_vendored_sam2_install(package_root):
    if package_root is None:
        raise ImportError(
            "Cannot find installed package 'sam2'. Install the vendored runtime with: "
            f"pip install -e {_VENDORED_SAM2_ROOT}"
        )

    expected_root = (_VENDORED_SAM2_ROOT / "sam2").resolve()
    try:
        package_root.relative_to(expected_root)
    except ValueError as exc:
        raise ImportError(
            "The active 'sam2' package is not the vendored sam_tracking runtime. "
            f"Found: {package_root}. Expected: {expected_root}. "
            f"Install it with: pip install -e {_VENDORED_SAM2_ROOT}"
        ) from exc


def _candidate_sam2_config_roots():
    env_root = os.environ.get("SAM2_CONFIG_ROOT")
    if env_root:
        yield Path(env_root)

    env_sam2_root = os.environ.get("SAM2_ROOT")
    if env_sam2_root:
        yield Path(env_sam2_root) / "configs"

    package_root = _installed_sam2_package_root()
    _require_vendored_sam2_install(package_root)
    yield package_root / "configs"


def _resolve_sam2_model_cfg():
    relative_path = Path("sam2") / "sam2_hiera_s.yaml"
    for config_root in _candidate_sam2_config_roots():
        candidate = config_root / relative_path
        if candidate.exists():
            return candidate
    return relative_path


sam2_args = {
    "sam_checkpoint": _as_str(_CKPT_ROOT / "sam2_hiera_small.pt"),
    "model_cfg": _as_str(_resolve_sam2_model_cfg()),
    "samurai_mode": True,
    "depth_score_weight": 0.7,
    "kf_score_weight": 0.15,
    "memory_bank_geo_score_threshold": 0.0,
}
