# SPDX-License-Identifier: MIT
"""Pure scalar checks shared by block-scale factories and kernel wrappers.

No Torch, Aiter, FlyDSL compiler or device initialization is needed here.
The wrappers retain their ValueError API; recipe factories translate only
these explicit parameter errors to an empty mapping.
"""


def m128_down_lds_bytes(*, n, k, topk, num_experts, persistent, num_oc_splits):
    if not isinstance(num_oc_splits, int) or num_oc_splits <= 0:
        raise ValueError("num_oc_splits must be a positive integer")
    if not isinstance(k, int) or k <= 0 or k % 256:
        raise ValueError("M128 requires K to be a positive integer multiple of256")
    if n <= 0 or n % (num_oc_splits * 128) or not 0 < topk <= min(num_experts, 255):
        raise ValueError(
            "M128 requires N divisible by num_oc_splits*128 and valid TOPK"
        )
    n_split = n // num_oc_splits
    scale_words = (n_split // 128 * 2 + 63 + 255) // 256 * 256
    arena_words = (
        (scale_words + 4 * 4096 + 128 + 128 * 3 + 2)
        if k == 256
        else (2 * 4096 + 128 * 2 + 2)
    ) + int(persistent)
    lds_bytes = arena_words * 4
    if lds_bytes > 160 * 1024:
        raise ValueError(
            f"M128 Down needs {lds_bytes} LDS bytes, exceeding gfx950's 163840"
        )
    return lds_bytes


def routed_down_lds_bytes(*, n, k, topk, num_experts, block_m, block_n, num_oc_splits):
    if not isinstance(k, int) or k <= 0 or k % 128:
        raise ValueError("8-wave Down requires K to be a positive multiple of128")
    if block_m != 256 or block_n not in (32, 64, 128):
        raise ValueError("8-wave Down requires M256 and N32/N64/N128 tiles")
    if not isinstance(num_oc_splits, int) or num_oc_splits <= 0:
        raise ValueError("num_oc_splits must be a positive integer")
    if n <= 0 or n % (num_oc_splits * 128) or n // num_oc_splits < 3 * block_n:
        raise ValueError(
            "8-wave Down requires N divisible by num_oc_splits*128 and at least 3 N tiles per split"
        )
    if not 0 < topk <= min(num_experts, 255):
        raise ValueError("8-wave Down requires 0 < TOPK <= min(num_experts,255)")
    scale_bytes = n // num_oc_splits // 128 * (k // 128) * 4
    lds_bytes = 4 * block_n * k + block_m * 8 + (scale_bytes + 15) // 16 * 16
    if lds_bytes > 160 * 1024:
        raise ValueError(
            f"8-wave Down needs {lds_bytes} LDS bytes, exceeding gfx950's 163840"
        )
    return lds_bytes
