"""Check a GPU host can run the Qwen extractor, before loading 28GB of weights.

Motivated by a real failure: on a host without CPython development headers,
Triton could not JIT-compile its CUDA helper, so every ``model.generate`` call
raised. Because :class:`~ocm.memory.write_pipeline.WritePipeline` absorbs
extraction failures as recorded validation failures (Req 3.3), the harness kept
running and would have reported a full arm table computed over an *empty* memory
store. This script surfaces that class of fault in seconds instead.

Usage::

    python -m ocm.scripts.gpu_doctor

Exits non-zero if anything required is missing. Checks are ordered cheapest
first, and each prints the remedy for its own failure.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile

_OK = "ok  "
_BAD = "FAIL"


def _line(status: str, name: str, detail: str) -> None:
    print(f"[{status}] {name}: {detail}", flush=True)


def check_python_headers() -> tuple[bool, str]:
    """Triton compiles a C helper against ``Python.h``; confirm it is present."""
    include = sysconfig.get_paths().get("include", "")
    header = os.path.join(include, "Python.h")
    if include and os.path.exists(header):
        return True, header
    return False, (
        f"{header or '<no include path>'} is missing.\n"
        f"       Install the headers for this interpreter "
        f"(python{sys.version_info.major}.{sys.version_info.minor}):\n"
        f"         sudo apt-get install -y python{sys.version_info.major}."
        f"{sys.version_info.minor}-dev\n"
        "       No root? Use an interpreter that ships headers, then reinstall:\n"
        f"         uv python install {sys.version_info.major}."
        f"{sys.version_info.minor}\n"
        f"         uv venv --python {sys.version_info.major}."
        f"{sys.version_info.minor} .venv && . .venv/bin/activate\n"
        "         pip install -r requirements.txt 'transformers>=4.45' accelerate"
    )


def check_compiler() -> tuple[bool, str]:
    """Triton shells out to a C compiler."""
    cc = shutil.which("gcc") or shutil.which("cc")
    if cc:
        return True, cc
    return False, "no gcc/cc on PATH. Install build-essential."


def check_triton_compile() -> tuple[bool, str]:
    """Actually build a trivial CPython extension the way Triton does.

    This is the direct analogue of the failing step, so it reproduces the fault
    without downloading a model.
    """
    cc = shutil.which("gcc") or shutil.which("cc")
    if not cc:
        return False, "skipped: no compiler"
    include = sysconfig.get_paths().get("include", "")
    source = "#include <Python.h>\nint probe(void) { return 0; }\n"
    with tempfile.TemporaryDirectory() as tmp:
        csrc = os.path.join(tmp, "probe.c")
        with open(csrc, "w", encoding="utf-8") as handle:
            handle.write(source)
        proc = subprocess.run(
            [cc, csrc, "-O0", "-shared", "-fPIC", "-o",
             os.path.join(tmp, "probe.so"), f"-I{include}"],
            capture_output=True,
            text=True,
        )
    if proc.returncode == 0:
        return True, "a CPython extension compiles"
    return False, (
        "compiling against Python.h failed, which is exactly what breaks Triton:\n"
        + "       " + (proc.stderr or "").strip().replace("\n", "\n       ")
    )


def check_torch_cuda() -> tuple[bool, str]:
    """Confirm torch sees a GPU."""
    try:
        import torch
    except ImportError as exc:
        return False, f"torch not installed ({exc}). Install the wheel for your CUDA."
    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() is False. Check drivers and CUDA_VISIBLE_DEVICES."
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    total = sum(
        torch.cuda.get_device_properties(i).total_memory
        for i in range(torch.cuda.device_count())
    )
    return True, f"torch {torch.__version__}, {len(names)}x {names[0]}, {total / 2**30:.0f}GiB total"


def check_gpu_kernel() -> tuple[bool, str]:
    """Run one real CUDA op, catching driver/kernel-image mismatches."""
    try:
        import torch
    except ImportError:
        return False, "skipped: torch not installed"
    if not torch.cuda.is_available():
        return False, "skipped: no CUDA device"
    try:
        a = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
        value = float((a @ a).float().sum())
    except Exception as exc:  # noqa: BLE001
        return False, f"a bf16 matmul on the GPU raised {exc!r}"
    return True, f"bf16 matmul ran (checksum {value:.1f})"


def check_transformers() -> tuple[bool, str]:
    try:
        import transformers
    except ImportError as exc:
        return False, f"not installed ({exc}). pip install 'transformers>=4.45' accelerate"
    return True, f"transformers {transformers.__version__}"


def check_capacity() -> tuple[bool, str]:
    """Qwen2.5-14B in bf16 needs roughly 30GiB of device memory."""
    try:
        import torch
    except ImportError:
        return False, "skipped: torch not installed"
    if not torch.cuda.is_available():
        return False, "skipped: no CUDA device"
    total = sum(
        torch.cuda.get_device_properties(i).total_memory
        for i in range(torch.cuda.device_count())
    ) / 2**30
    need = 30.0
    if total >= need:
        return True, f"{total:.0f}GiB available, ~{need:.0f}GiB needed for 14B bf16"
    return False, (
        f"only {total:.0f}GiB across all devices but ~{need:.0f}GiB is needed for "
        "Qwen2.5-14B in bf16. Use a smaller model or add devices; note that "
        "changing the model breaks comparability with OCMR's published table."
    )


CHECKS = (
    ("python headers", check_python_headers),
    ("c compiler", check_compiler),
    ("cpython extension build", check_triton_compile),
    ("transformers", check_transformers),
    ("torch + cuda", check_torch_cuda),
    ("gpu kernel", check_gpu_kernel),
    ("device capacity", check_capacity),
)


def main() -> int:
    """Run every check; return 1 if any failed."""
    print("Checking this host can run the Qwen extraction arm.\n", flush=True)
    failed = []
    for name, check in CHECKS:
        try:
            ok, detail = check()
        except Exception as exc:  # noqa: BLE001 - a check must never crash the doctor
            ok, detail = False, f"check itself raised {exc!r}"
        _line(_OK if ok else _BAD, name, detail)
        if not ok:
            failed.append(name)

    if failed:
        print(
            f"\n{len(failed)} check(s) failed: {', '.join(failed)}.\n"
            "Do not start the sweep yet. Extraction failures are absorbed by the "
            "write pipeline, so a run in this state produces a complete but "
            "meaningless table.",
            flush=True,
        )
        return 1
    print(
        "\nAll checks passed. Next:\n"
        "  python -m ocm.evaluation.rahgm.run_ocmr_arm --extractor qwen "
        "--per-category 2 --seeds 1337 --no-write",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
