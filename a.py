def cast_to_representable(inp, scale = 1., fp8_format='e4m3'):
    import transformer_engine.pytorch.cpp_extensions as texcpp
    import transformer_engine_extensions as tex
    fp8_type = tex.DType.kFloat8E4M3 if fp8_format == 'e4m3' else tex.DType.kFloat8E5M2
    input_type = texcpp.TE_DType[inp.dtype]
    meta = tex.FP8TensorMeta()
    meta.scale = torch.ones(1,dtype=torch.float32, device="cuda") * scale
    meta.scale_inv = torch.ones(1, dtype=torch.float32, device="cuda") / scale
    meta.amax_history = torch.zeros(1, 1, dtype=torch.float32, device="cuda")
    ret = texcpp.cast_to_fp8(inp, meta, tex.FP8FwdTensors.GEMM1_INPUT, fp8_type)
    ret = texcpp.cast_from_fp8(ret, meta, tex.FP8FwdTensors.GEMM1_INPUT, fp8_type, input_type)
    return ret


import torch
cast_to_representable(torch.randn(3, 3))

