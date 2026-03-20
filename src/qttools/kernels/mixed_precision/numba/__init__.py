# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.


def compress(*args, **kwargs):
    raise NotImplementedError(
        "Compression is not implemented for the current array module."
    )


def decompress(*args, **kwargs):
    raise NotImplementedError(
        "Decompression is not implemented for the current array module."
    )


__all__ = ["compress", "decompress"]
