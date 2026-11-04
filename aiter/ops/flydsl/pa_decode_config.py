"""Calibrated PA defaults used when the native tuning cache has no entry."""

# gfx942 MiMo Flash Q8: (partition cap, query splits). These are geometry
# defaults from the validated 4K--1M serving matrix, not per-token retuning.
_MIMO_Q8_DEFAULTS = {
    12: (24, 1),
    24: (64, 2),
    40: (64, 2),
    96: (3, 1),
    152: (2, 1),
    208: (8, 2),
    240: (4, 2),
    256: (4, 2),
}


def validate_pa_config(values, *, max_partitions, query_length, requested_query_splits=None):
    """Validate persisted tuning values before allocating a plan/workspace."""
    budget = values.get("workgroup_budget")
    partitions = values.get("partition_count", max_partitions)
    splits = values.get("query_splits", requested_query_splits)
    if (
        not {"workgroup_budget"} <= set(values) <= {"workgroup_budget", "partition_count", "query_splits"}
        or type(budget) is not int or budget < 1
        or type(partitions) is not int or not 1 <= partitions <= max_partitions
        or (splits is not None and (type(splits) is not int or splits < 1 or query_length % splits))
        or (requested_query_splits is not None and splits != requested_query_splits)
    ):
        raise ValueError("Invalid PA partition/query configuration in FlyDSL autotune cache")
    return budget, partitions, splits


def calibrated_pa_config(*, arch, num_seqs, num_kv_heads, query_group_size,
                         head_dim, value_dim, block_size, query_length,
                         query_dtype, kv_dtype, per_token_kv, trans_v,
                         sink_dtype, sliding_window, max_partitions, num_cu,
                         requested_query_splits=None, **unused):
    """Return a validated default, or None to preserve the generic policy."""
    if (arch != "gfx942" or num_cu != 304 or num_kv_heads != 1
            or query_group_size != 16 or (head_dim, value_dim, block_size, query_length) != (192, 128, 64, 8)
            or query_dtype != "torch.bfloat16" or kv_dtype != "torch.bfloat16"
            or per_token_kv or not trans_v or sink_dtype is not None or sliding_window):
        return None
    selected = _MIMO_Q8_DEFAULTS.get(num_seqs)
    if selected is None or selected[0] > max_partitions:
        # Preserve a full plan budget for other batches of this geometry.
        # A sparse generic budget would disable the repaired query-group grid.
        partitions, splits = min(16, max_partitions), None
    else:
        partitions, splits = selected
    if requested_query_splits is not None:
        splits = requested_query_splits
    return dict(workgroup_budget=num_seqs * num_kv_heads * partitions,
                partition_count=partitions, query_splits=splits)
