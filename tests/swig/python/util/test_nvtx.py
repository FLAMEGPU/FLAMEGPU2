import pytest
from unittest import TestCase
from pyflamegpu import *

class NVTXTest(TestCase):

    def test_nvtx(self):

        # Assert that the Enabled flag can be accessed, and that it is a bool.
        is_enabled = pyflamegpu.NVTX_ENABLED
        assert type(is_enabled) is bool 
        print(is_enabled)

        # Call each public method, but they do not have any side effects that make them observable (which would also require it to be enabled). They should not thorw however?
        try:
            pyflamegpu.nvtx_push("test")
        except Exception as e:
            assert False, f"NVTX Push raised an exception {e}"

        try:
            pyflamegpu.nvtx_pop()
        except Exception as e:
            assert False, f"NVTX pop raised an exception {e}"

        # Python doesn't have nvtx ranges in case of GC related delays in dtoring. push/pop must be used.
        with pytest.raises(AttributeError):
            pyflamegpu.nvtx_range("test_range")
        with pytest.raises(AttributeError):
            pyflamegpu.nvtx_Range("test_range")