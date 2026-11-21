"""Process-tree and OS memory observations, independent of the MLX allocator."""

import ctypes
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
import platform
from time import monotonic_ns

import psutil

SAMPLING_MS = 50


@lru_cache
def _libproc():
    lib = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    call = lib.proc_pid_rusage
    call.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    call.restype = ctypes.c_int
    return call


def mac_footprint(pid: int) -> int | None:
    if platform.system() != "Darwin":
        return None
    # Darwin <sys/resource.h> rusage_info_v4 prefix through ri_phys_footprint.
    # Allocate slack for the rest of the fixed v4 ABI.
    data = (ctypes.c_uint64 * 128)()
    try:
        if _libproc()(pid, 4, ctypes.byref(data)) == 0:
            return data[9]
    except (OSError, AttributeError):
        pass
    return None


@lru_cache
def _sysctl():
    call = ctypes.CDLL(None, use_errno=True).sysctlbyname
    call.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
                     ctypes.c_void_p, ctypes.c_size_t]
    call.restype = ctypes.c_int
    return call


def system_pressure() -> str:
    if platform.system() == "Darwin":
        result = ctypes.c_int()
        size = ctypes.c_size_t(ctypes.sizeof(result))
        if _sysctl()(b"kern.memorystatus_vm_pressure_level", ctypes.byref(result),
                     ctypes.byref(size), None, 0) == 0:
            return {1: "normal", 2: "warning", 4: "critical"}.get(result.value, "unknown")
    return "unknown"


def environment(mx=None) -> dict:
    packages = {}
    for name in ("mlx-spec-bench", "mlx", "mlx-lm", "transformers"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "1.0.0" if name == "mlx-spec-bench" else None
    info = mx.device_info() if mx else {}
    return {
        "platform": platform.system(), "os_release": platform.mac_ver()[0] or platform.release(),
        "architecture": platform.machine(), "device": info.get("device_name") or platform.processor()
        or platform.machine(), "backend": str(mx.default_device()) if mx else "uninitialized",
        "memory_bytes": psutil.virtual_memory().total, "python": platform.python_version(),
        "packages": packages, "hostname": platform.node() or "unknown", "logical_cpus": psutil.cpu_count(),
    }


class Sampler:
    def __init__(self, pid: int | None, parent_pid: int | None = None):
        self.pid = pid
        self.parent_pid = parent_pid
        self.consecutive_pressure = 0
        self.pressure_start_ns = None
        self.observations = {
            "sampling_period_ms": SAMPLING_MS, "sample_count": 0,
            "peak_process_rss_bytes": None, "peak_process_footprint_bytes": None,
            "footprint_samples": 0, "system_available_min_bytes": None,
            "system_swap_start_bytes": None, "system_swap_end_bytes": None, "system_swap_peak_bytes": None,
            "pressure_samples": {key: 0 for key in ("normal", "warning", "critical", "unknown")},
            "pressure_max_consecutive_elevated_samples": 0, "pressure_max_elevated_span_ns": 0,
        }

    def sample(self) -> int | None:
        data = self.observations
        data["sample_count"] += 1
        available = psutil.virtual_memory().available
        swap = psutil.swap_memory().used
        previous_available = data["system_available_min_bytes"]
        data["system_available_min_bytes"] = available if previous_available is None else min(previous_available, available)
        if data["system_swap_start_bytes"] is None:
            data["system_swap_start_bytes"] = swap
        data["system_swap_end_bytes"] = swap
        data["system_swap_peak_bytes"] = max(data["system_swap_peak_bytes"] or 0, swap)
        pressure = system_pressure()
        data["pressure_samples"][pressure] += 1
        if pressure in ("warning", "critical"):
            self.consecutive_pressure += 1
            now = monotonic_ns()
            if self.pressure_start_ns is None:
                self.pressure_start_ns = now
            data["pressure_max_elevated_span_ns"] = max(
                data["pressure_max_elevated_span_ns"], now - self.pressure_start_ns
            )
        else:
            self.consecutive_pressure = 0
            self.pressure_start_ns = None
        data["pressure_max_consecutive_elevated_samples"] = max(
            data["pressure_max_consecutive_elevated_samples"], self.consecutive_pressure
        )
        if self.pid is None:
            return None
        try:
            root = psutil.Process(self.pid)
            processes = [root, *root.children(recursive=True)]
            if self.parent_pid is not None:
                processes.append(psutil.Process(self.parent_pid))
            rss = sum(p.memory_info().rss for p in processes)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None
        if rss > 0:
            data["peak_process_rss_bytes"] = max(data["peak_process_rss_bytes"] or 0, rss)
        footprints = [mac_footprint(p.pid) for p in processes]
        if all(value is not None and value > 0 for value in footprints):
            footprint = sum(footprints)
            data["footprint_samples"] += 1
            data["peak_process_footprint_bytes"] = max(data["peak_process_footprint_bytes"] or 0, footprint)
            return footprint
        return rss or None
