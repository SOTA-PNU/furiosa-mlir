"""numpy buffers for graph tensors: dtype names, safetensors files, and MLIR memref descriptors."""

import ctypes
import json
import struct
from pathlib import Path

import ml_dtypes
import numpy as np

# Graph dtype -> numpy dtype.
DTYPES = {
    "fp32": np.dtype(np.float32),
    "fp16": np.dtype(np.float16),
    "bf16": np.dtype(ml_dtypes.bfloat16),
    "i64": np.dtype(np.int64),
    "i32": np.dtype(np.int32),
    "i16": np.dtype(np.int16),
    "i8": np.dtype(np.int8),
    "u8": np.dtype(np.uint8),
    "fp8e4nv": np.dtype(ml_dtypes.float8_e4m3fn),
}

# safetensors dtype tag <-> numpy dtype.
_TAGS = {"F64": np.float64, "F32": np.float32, "F16": np.float16, "BF16": ml_dtypes.bfloat16, "I64": np.int64,
         "I32": np.int32, "I16": np.int16, "I8": np.int8, "U8": np.uint8, "BOOL": np.bool_,
         "F8_E4M3": ml_dtypes.float8_e4m3fn}
_TAG_OF = {np.dtype(v): k for k, v in _TAGS.items()}


def load(path: Path) -> dict[str, np.ndarray]:
    """Every tensor of a safetensors file, as writable C-contiguous arrays."""
    data = Path(path).read_bytes()
    (size,) = struct.unpack("<Q", data[:8])
    header = json.loads(data[8 : 8 + size])
    body = memoryview(data)[8 + size :]
    tensors = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        begin, end = info["data_offsets"]
        array = np.frombuffer(body[begin:end], dtype=_TAGS[info["dtype"]]).reshape(info["shape"])
        tensors[name] = array.copy()
    return tensors


def save(path: Path, tensors: dict[str, np.ndarray]) -> None:
    header, chunks, offset = {}, [], 0
    for name, array in tensors.items():
        raw = np.ascontiguousarray(array).tobytes()
        header[name] = {"dtype": _TAG_OF[array.dtype], "shape": list(array.shape), "data_offsets": [offset, offset + len(raw)]}
        chunks.append(raw)
        offset += len(raw)
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    Path(path).write_bytes(struct.pack("<Q", len(blob)) + blob + b"".join(chunks))


_DESCRIPTORS = {}


def descriptor_type(rank: int):
    """The C struct of an MLIR strided memref of ``rank`` (identity layout)."""
    if rank not in _DESCRIPTORS:
        fields = [("allocated", ctypes.c_void_p), ("aligned", ctypes.c_void_p), ("offset", ctypes.c_int64)]
        if rank:
            fields += [("sizes", ctypes.c_int64 * rank), ("strides", ctypes.c_int64 * rank)]
        _DESCRIPTORS[rank] = type(f"MemRefDescriptor{rank}", (ctypes.Structure,), {"_fields_": fields})
    return _DESCRIPTORS[rank]


def descriptor(array: np.ndarray):
    """A memref descriptor viewing ``array``, which must be C-contiguous and outlive the call."""
    assert array.flags.c_contiguous, "memref arguments must be C-contiguous"
    desc = descriptor_type(array.ndim)()
    desc.allocated = desc.aligned = array.ctypes.data
    desc.offset = 0
    if array.ndim:
        desc.sizes[:] = array.shape
        desc.strides[:] = [s // array.itemsize for s in array.strides]
    return desc


def call(address: int, arrays: list[np.ndarray]) -> None:
    """Calls a ``_mlir_ciface_`` function whose every argument is a memref, in order."""
    descs = [descriptor(a) for a in arrays]
    prototype = ctypes.CFUNCTYPE(None, *[ctypes.POINTER(type(d)) for d in descs])
    prototype(address)(*[ctypes.byref(d) for d in descs])
