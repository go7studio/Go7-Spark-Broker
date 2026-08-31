from __future__ import annotations

import ctypes
import json
from typing import Any


class CudaMemoryProbeError(RuntimeError):
    pass


def _call(function: Any, name: str, *args: Any) -> None:
    code = int(function(*args))
    if code != 0:
        raise CudaMemoryProbeError(f"{name} failed with CUDA status {code}")


def snapshot() -> dict[str, int]:
    """Measure memory available to a new short-lived CUDA process context.

    This helper is intentionally run in a short-lived child process by the
    resource probe. Keeping the CUDA context out of the long-lived inventory
    service prevents the observer itself from becoming a persistent GPU
    consumer.
    """

    try:
        driver = ctypes.CDLL("libcuda.so.1")
    except OSError as exc:
        raise CudaMemoryProbeError("CUDA driver library is unavailable") from exc

    cu_init = driver.cuInit
    cu_device_get = driver.cuDeviceGet
    cu_primary_retain = driver.cuDevicePrimaryCtxRetain
    cu_primary_release = driver.cuDevicePrimaryCtxRelease
    cu_context_set = driver.cuCtxSetCurrent
    cu_mem_get_info = getattr(driver, "cuMemGetInfo_v2", None) or getattr(
        driver, "cuMemGetInfo", None
    )
    if cu_mem_get_info is None:
        raise CudaMemoryProbeError("CUDA memory query entry point is unavailable")

    for function in (
        cu_init,
        cu_device_get,
        cu_primary_retain,
        cu_primary_release,
        cu_context_set,
        cu_mem_get_info,
    ):
        function.restype = ctypes.c_int
    cu_init.argtypes = [ctypes.c_uint]
    cu_device_get.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    cu_primary_retain.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
    cu_primary_release.argtypes = [ctypes.c_int]
    cu_context_set.argtypes = [ctypes.c_void_p]
    cu_mem_get_info.argtypes = [
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
    ]

    _call(cu_init, "cuInit", 0)
    device = ctypes.c_int()
    _call(cu_device_get, "cuDeviceGet", ctypes.byref(device), 0)
    context = ctypes.c_void_p()
    retained = False
    try:
        _call(
            cu_primary_retain,
            "cuDevicePrimaryCtxRetain",
            ctypes.byref(context),
            device,
        )
        retained = True
        _call(cu_context_set, "cuCtxSetCurrent", context)
        free_bytes = ctypes.c_size_t()
        total_bytes = ctypes.c_size_t()
        _call(
            cu_mem_get_info,
            "cuMemGetInfo",
            ctypes.byref(free_bytes),
            ctypes.byref(total_bytes),
        )
    finally:
        if retained:
            cu_context_set(ctypes.c_void_p())
            cu_primary_release(device)

    if total_bytes.value <= 0 or free_bytes.value > total_bytes.value:
        raise CudaMemoryProbeError("CUDA driver returned an invalid memory envelope")
    return {
        "allocatableBytes": int(free_bytes.value),
        "addressSpaceTotalBytes": int(total_bytes.value),
    }


def main() -> None:
    print(json.dumps(snapshot(), separators=(",", ":"), sort_keys=True))


if __name__ == "__main__":
    main()
