try:
    import turbo
except ImportError:
    turbo = None

def turbo_is_available():
    return turbo is not None

def assert_turbo_is_available():
    assert turbo_is_available(), (
        'Turbo not available. Please run `pip install mosaicml-turbo==0.0.4`.')

quantize_signed = turbo.quantize_signed if turbo_is_available() else None
dequantize_signed = turbo.dequantize_signed if turbo_is_available() else None
ElemwiseOps = turbo.ElemwiseOps if turbo_is_available() else None
