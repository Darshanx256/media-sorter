from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
import importlib
import importlib.util
import sys


@dataclass(slots=True)
class DependencyStatus:
    name: str
    ok: bool
    version: str | None
    detail: str
    install_hint: str


DEPENDENCY_HINTS: dict[str, str] = {
    "torch": "Install torch for your CPU or CUDA target, then install open-clip-torch.",
    "torchvision": "Reinstall a matched torch/torchvision build if torchvision is broken.",
    "open_clip": "Install open-clip-torch after torch is working.",
    "clip": "Optional fallback backend. Install clip-anytorch only if you need the compatibility path.",
    "onnx": "Install finalize extras with: pip install -e .[finalize]",
    "onnxruntime": "Install finalize extras with: pip install -e .[finalize]",
    "onnxscript": "Install finalize extras with: pip install -e .[finalize]",
    "cv2": "Install video extras with: pip install -e .[video]",
}


def collect_dependency_status() -> list[DependencyStatus]:
    statuses: list[DependencyStatus] = []
    for module_name in ("torch", "torchvision", "open_clip", "clip", "onnx", "onnxruntime", "onnxscript", "cv2"):
        try:
            if module_name == "clip":
                spec = importlib.util.find_spec("clip")
                if spec is None:
                    raise ModuleNotFoundError("No module named 'clip'")
                version = _dist_version_for(module_name)
            else:
                module = importlib.import_module(module_name)
                version = getattr(module, "__version__", None) or _dist_version_for(module_name)
            statuses.append(DependencyStatus(module_name, True, version, "ok", DEPENDENCY_HINTS[module_name]))
        except Exception as exc:
            statuses.append(
                DependencyStatus(
                    module_name,
                    False,
                    None,
                    f"{type(exc).__name__}: {exc}",
                    DEPENDENCY_HINTS[module_name],
                )
            )
    return statuses


def render_doctor_report(*, expect_video: bool = False, expect_finalize: bool = False) -> str:
    statuses = collect_dependency_status()
    by_name = {status.name: status for status in statuses}

    lines = [
        "media-sorter environment report",
        f"python_executable: {sys.executable}",
        f"python_version: {sys.version.split()[0]}",
        "",
    ]

    for status in statuses:
        if status.ok:
            version = status.version or "unknown"
            lines.append(f"{status.name}: ok ({version})")
        else:
            lines.append(f"{status.name}: missing_or_broken ({status.detail})")

    lines.extend(["", "feature_readiness:"])
    lines.append(
        f"ml_backend: {_feature_status(by_name, ('torch', 'open_clip'), optional=('clip',))}"
    )
    lines.append(
        f"finalize_bundle: {_feature_status(by_name, ('torch', 'open_clip', 'onnx', 'onnxruntime', 'onnxscript'))}"
    )
    lines.append(f"video_runtime: {_feature_status(by_name, ('cv2',))}")

    lines.extend(["", "recommended_checks:"])
    for module_name in _recommended_modules(expect_video=expect_video, expect_finalize=expect_finalize):
        status = by_name[module_name]
        if not status.ok:
            lines.append(f"- {module_name}: {status.install_hint}")
    if lines[-1] == "recommended_checks:":
        lines.append("- environment looks healthy for the requested feature set")

    return "\n".join(lines) + "\n"


def _dist_version_for(module_name: str) -> str | None:
    names = {
        "open_clip": "open-clip-torch",
        "clip": "clip-anytorch",
        "cv2": "opencv-python-headless",
    }
    try:
        return metadata.version(names.get(module_name, module_name))
    except metadata.PackageNotFoundError:
        return None


def _feature_status(
    statuses: dict[str, DependencyStatus],
    required: tuple[str, ...],
    optional: tuple[str, ...] = (),
) -> str:
    missing_required = [name for name in required if not statuses[name].ok]
    if missing_required:
        return "missing " + ", ".join(missing_required)
    missing_optional = [name for name in optional if not statuses[name].ok]
    if missing_optional:
        return "ready (optional gaps: " + ", ".join(missing_optional) + ")"
    return "ready"


def _recommended_modules(*, expect_video: bool, expect_finalize: bool) -> list[str]:
    modules = ["torch", "torchvision", "open_clip"]
    if expect_finalize:
        modules.extend(["onnx", "onnxruntime", "onnxscript"])
    if expect_video:
        modules.append("cv2")
    return modules
