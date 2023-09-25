try:
    import grouped_gemm
except ImportError:
    grouped_gemm = None

def grouped_gemm_is_available():
    return grouped_gemm is not None

def assert_grouped_gemm_is_available():
    assert grouped_gemm_is_available(), (
        'Turbo not available. Please run `pip install mosaicml-turbo==0.0.4`.')

gmm = grouped_gemm.ops.gmm if grouped_gemm_is_available() else None
