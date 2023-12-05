      fp8_recipe = DelayedScaling(**precision_config)
        fp16_context = torch.autocast(device_type='cuda', dtype=torch.float16) if self.args.fp16 or self.args.bf16 else contextlib.nullcontext()
        for i, size in enumerate(batch_sizes):
            weights = b[i]
            with fp16_context:
                with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                    pad = (0, 0, 0, 8 - size % 8)
                    padded_tensor = F.pad(a[start:start + size, :], pad, "constant", 0)
                    res = weights()
            out.append(res)
            start += size
        return torch.cat(ou