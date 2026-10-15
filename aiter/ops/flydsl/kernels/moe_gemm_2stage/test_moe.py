import torch

# 解压缩权重，得到用于参考计算的浮点表示
def _reference_weight(weight, scale, expert, kind, interleave, gate):
    value = weight[expert]
    n, k = value.shape
    shuffled = bool(getattr(weight, "is_shuffled", False))
    if kind == "fp4":
        value = value.view(torch.uint8)
    if shuffled and interleave and gate:
        value = value.view(n // 32, 2, k // 64, 4, 16, 16)
        value = value.permute(1, 0, 4, 2, 3, 5).reshape(n, k)
    elif shuffled:
        pack = 16 // value.element_size()
        value = value.view(n // 16, k // pack, 16, pack)
        value = value.permute(0, 2, 1, 3).reshape(n, k)
    if kind == "bf16":
        return value
    if kind == "fp4":
        from aiter.utility import fp4_utils

        cols = k * 2 // 32
        if shuffled:
            padded = (cols + 7) // 8 * 8
            scales = scale.view(torch.uint8).reshape(-1)[expert * n * padded:(expert + 1) * n * padded]
            scales = scales.view(n // 32, padded // 8, 4, 16, 2, 2)
            order = (5, 0, 3, 1, 4, 2) if interleave and gate else (0, 5, 3, 1, 4, 2)
            scales = scales.permute(order).reshape(n, padded)[:, :cols]
        else:
            scales = scale.view(torch.uint8).reshape(weight.shape[0], n, cols)[expert]
        value = fp4_utils.mxfp4_to_f32(value.view(torch.float4_e2m1fn_x2)).view(n, cols, 32)
        return (value * fp4_utils.e8m0_to_f32(scales).unsqueeze(-1)).reshape(n, k * 2).to(torch.bfloat16)
    value = value.float()
    if kind == "ptpc":
        value = value * scale.reshape(weight.shape[0], n, 1)[expert]
    elif kind == "per_tensor":
        value = value * scale.reshape(-1)[expert]
    else:
        value = (value.view(n // 128, 128, k // 128, 128)
                 * scale.reshape(weight.shape[0], n // 128, k // 128)[expert, :, None, :, None])
    value = value.reshape(n, k)
    return value if kind == "block" else value.to(torch.bfloat16)

# 按照要求的kind量化+反量化激活，模拟量化精度损失
def _reference_activation(value, kind, quant_dtype):
    if kind in ["", "bf16"] or kind is None:
        return value

    if kind == "a4w4":
        import aiter
        from aiter.utility import fp4_utils

        quant, scales = aiter.get_torch_quant(aiter.QuantType.per_1x32)(value, quant_dtype=quant_dtype)
        dequant = (fp4_utils.mxfp4_to_f32(quant).view(-1, 32)
                   * fp4_utils.e8m0_to_f32(scales).view(-1, 1))
        return dequant.reshape(value.shape).to(value.dtype)

    assert kind in ("ptpc", "per_tensor", "block")
    shape = value.shape
    data = value.float().reshape(-1, 128 if kind == "block" else shape[-1])
    amax = data.abs().amax() if kind == "per_tensor" else data.abs().amax(dim=-1, keepdim=True)
    scale = amax / torch.finfo(quant_dtype).max
    scale = torch.where(scale == 0, 1.0, scale)
    result = ((data / scale).to(quant_dtype).float() * scale).reshape(shape)
    return result if kind == "block" else result.to(value.dtype)



def torch_reference(hidden_states, w1, w2,  *, mxfp4_activations=False):
    """独立 Torch MoE 参考；沿用公共 API 默认值、原量化语义和数值顺序。

    只在调用时导入算子以补参数和验证输入，不执行任何被测 kernel。
    FP8 对输入和中间结果量化；MXFP4 使用 BF16 激活；GELU 为非 gated。
    固定 A4W4 kernel 测试可显式模拟两次 MXFP4 激活量化；API 调优仍用默认参考。
    """
    import aiter
    import torch
    from aiter.ops.flydsl.moe_common import GateMode
    from pyhip.ops.moe.tuned_moe import _native_kind, fused_moe

    bound = inspect.signature(fused_moe).bind(**call)
    bound.apply_defaults()
    call = bound.arguments
    x, w1, w2, ids, weights = (call[name] for name in
                              ("hidden_states", "w1", "w2", "topk_ids", "topk_weight"))
    kind = _native_kind(call)
    if kind is None:
        raise ValueError("independent MoE validation is unavailable for these inputs")
    if mxfp4_activations and kind != "fp4":
        raise ValueError("MXFP4 activation reference requires MXFP4 weights")
    act_kind = "a4w4" if mxfp4_activations else kind
    if not ((ids >= 0) & (ids < w1.shape[0])).all().item():
        raise ValueError("topk_ids contains an out-of-range expert")
    b, h = x.shape
    gelu = call["activation"] == aiter.ActivationType.Gelu
    i, topk = w1.shape[1] // (1 if gelu else 2), ids.shape[1]
    interleave = call["gate_mode"] == GateMode.INTERLEAVE
    dtype = x.dtype
    x = _reference_activation(x, act_kind, w1.dtype)
    mid = torch.empty((b, topk, i), dtype=dtype, device=x.device)
    for expert in range(w1.shape[0]):
        row, slot = torch.where(ids == expert)
        if row.numel() == 0:
            continue
        weight = _reference_weight(w1, call["w1_scale"], expert, kind, interleave, True)
        projection = x[row].float() @ weight.float().T
        act = call["activation"]
        if gelu:
            value = torch.nn.functional.gelu(projection)
        else:
            gate, up = projection.chunk(2, dim=-1)
            if act == aiter.ActivationType.Swiglu:
                limit = 7.0 if call["swiglu_limit"] is None else float(call["swiglu_limit"])
                gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
                value = gate * torch.sigmoid(1.702 * gate) * (up + 1.0)
            elif act == aiter.ActivationType.Situv2:
                beta = 1.0 if call["beta"] is None else float(call["beta"])
                linear_beta = 1.0 if call["linear_beta"] is None else float(call["linear_beta"])
                value = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
                value = value * (linear_beta * torch.tanh(up / linear_beta))
            else:
                value = torch.nn.functional.silu(gate) * up
        mid[row, slot] = value.to(dtype)
    mid = _reference_activation(mid, act_kind, w1.dtype)
    result = torch.zeros((b, h), dtype=torch.float32, device=x.device)
    for expert in range(w2.shape[0]):
        row, slot = torch.where(ids == expert)
        if row.numel() == 0:
            continue
        weight = _reference_weight(w2, call["w2_scale"], expert, kind, interleave, False)
        value = (mid[row, slot].float() @ weight.float().T) * weights[row, slot, None]
        result.index_add_(0, row, value)
    return result.to(dtype)
