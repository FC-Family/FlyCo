# ⭐⭐⭐******************************************************************⭐⭐⭐
# Authors      :    Chen Feng <cfengag at connect dot ust dot hk>, UAV Group, ECE, HKUST.
#                   Guiyong Zheng <shverses at gmail dot com>, SAI, SYSU.
# Homepage     :    https://chen-albert-feng.github.io/AlbertFeng.github.io/
#                   https://gy920.github.io/
# Date         :    Jun. 2026
# E-mail       :    cfengag at connect dot ust dot hk
#                   shverses at gmail dot com
# Description  :    This file is part of the FlyCo Perception public ROS runtime.
# Copyright    :    Copyright (c) 2026 Chen Feng and Guiyong Zheng.
# License      :    PolyForm Noncommercial License 1.0.0
#                   <https://polyformproject.org/licenses/noncommercial/1.0.0/>
# Project      :    FlyCo: Foundation Model-Empowered Drones for Autonomous 3D Structure Scanning in Open-World Environments
# Website      :    https://hkust-aerial-robotics.github.io/FC-Planner/
# ⭐⭐⭐******************************************************************⭐⭐⭐

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf


HF_ID_PREFIXES = ("facebook/", "google/", "openai/")


@dataclass
class OfficialRuntimeSpec:
    """Resolved resource bundle for the official runtime path."""

    runtime_config_path: str
    model_config_path: str
    camera_param_default: Optional[str]
    mesh_pred_checkpoint: Optional[str]
    nksr_checkpoint: Optional[str]
    dino_model: str
    siglip_model: str


def discover_package_root(path_hint: Optional[str] = None) -> Path:
    search_roots = []
    if path_hint:
        hint_path = Path(path_hint).resolve()
        search_roots.append(hint_path if hint_path.is_dir() else hint_path.parent)

    search_roots.append(Path(__file__).resolve().parent)

    for root in search_roots:
        for candidate in (root, *root.parents):
            if (candidate / "package.xml").exists():
                return candidate

    try:
        import rospkg

        return Path(rospkg.RosPack().get_path("mesh_predict"))
    except Exception as exc:
        raise RuntimeError(
            "mesh_predict package root could not be resolved from the current "
            "runtime context."
        ) from exc


def resolve_package_path(
    value: Optional[str], package_root: Optional[Path] = None
) -> Optional[Path]:
    if value is None or value == "":
        return None

    path = Path(value)
    if path.is_absolute():
        return path
    root = package_root or discover_package_root()
    return root / path


def resolve_existing_file(
    label: str, value: Optional[str], package_root: Optional[Path] = None
) -> str:
    path = resolve_package_path(value, package_root=package_root)
    if path is None or not path.exists():
        raise RuntimeError(
            f"mesh_predict runtime resource '{label}' is missing. "
            f"Expected file: {path if path is not None else value}"
        )
    return str(path)


def resolve_path_or_id(
    value: Optional[str],
    env_name: Optional[str],
    package_root: Optional[Path] = None,
) -> str:
    env_value = os.environ.get(env_name, "").strip() if env_name else ""
    candidate = env_value or (value or "")
    if not candidate:
        raise RuntimeError(
            f"mesh_predict runtime resource '{env_name or 'resource'}' is not set."
        )

    path = resolve_package_path(candidate, package_root=package_root)
    if path is not None and path.exists():
        return str(path)

    if candidate.startswith(HF_ID_PREFIXES):
        return candidate

    return candidate


def resolve_required_path_from_entry(
    entry: dict, package_root: Optional[Path] = None
) -> str:
    env_name = entry.get("env")
    env_value = os.environ.get(env_name, "").strip() if env_name else ""
    candidate = env_value or entry.get("default")
    return resolve_existing_file(
        env_name or "resource", candidate, package_root=package_root
    )


def resolve_path_or_id_from_entry(
    entry: dict, package_root: Optional[Path] = None
) -> str:
    return resolve_path_or_id(
        entry.get("default"), entry.get("env"), package_root=package_root
    )


def is_runtime_config(conf) -> bool:
    return "runtime" in conf and "resources" in conf


def load_runtime_yaml(runtime_config_path: str):
    resolved_path = resolve_existing_file("runtime_config_path", runtime_config_path)
    return OmegaConf.to_container(OmegaConf.load(resolved_path), resolve=True)


def load_official_runtime_spec(
    runtime_config_path: str, checkpoint_override: Optional[str] = None
) -> OfficialRuntimeSpec:
    resolved_runtime_config_path = resolve_existing_file(
        "runtime_config_path", runtime_config_path
    )
    package_root = discover_package_root(resolved_runtime_config_path)
    conf = load_runtime_yaml(resolved_runtime_config_path)

    if not is_runtime_config(conf):
        checkpoint_path = None
        if checkpoint_override:
            checkpoint_path = resolve_existing_file(
                "mesh_pred_ckpt_path",
                checkpoint_override,
                package_root=package_root,
            )
        return OfficialRuntimeSpec(
            runtime_config_path=resolved_runtime_config_path,
            model_config_path=resolved_runtime_config_path,
            camera_param_default=None,
            mesh_pred_checkpoint=checkpoint_path,
            nksr_checkpoint=None,
            dino_model="facebook/dinov2-base",
            siglip_model="google/siglip-base-patch16-224",
        )

    runtime = conf["runtime"]
    resources = conf["resources"]

    if checkpoint_override:
        mesh_pred_checkpoint = resolve_existing_file(
            "mesh_pred_ckpt_path",
            checkpoint_override,
            package_root=package_root,
        )
    else:
        mesh_pred_checkpoint = resolve_required_path_from_entry(
            resources["mesh_pred_checkpoint"], package_root=package_root
        )

    camera_param_default = runtime.get("camera_param_default")
    if camera_param_default:
        camera_param_default = resolve_existing_file(
            "camera_param_default",
            camera_param_default,
            package_root=package_root,
        )

    return OfficialRuntimeSpec(
        runtime_config_path=resolved_runtime_config_path,
        model_config_path=resolve_existing_file(
            "model_config_path",
            runtime["model_config_path"],
            package_root=package_root,
        ),
        camera_param_default=camera_param_default,
        mesh_pred_checkpoint=mesh_pred_checkpoint,
        nksr_checkpoint=resolve_required_path_from_entry(
            resources["nksr_checkpoint"], package_root=package_root
        ),
        dino_model=resolve_path_or_id_from_entry(
            resources["dino_model"], package_root=package_root
        ),
        siglip_model=resolve_path_or_id_from_entry(
            resources["siglip_model"], package_root=package_root
        ),
    )


def load_optional_yaml(config_path: Optional[str]) -> dict:
    if not config_path:
        return {}
    resolved_path = resolve_existing_file("profile_config_path", config_path)
    return OmegaConf.to_container(OmegaConf.load(resolved_path), resolve=True)
