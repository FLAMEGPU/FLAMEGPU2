import pytest
import os
from unittest import TestCase
from pyflamegpu import *
import sys

class EnvironmentTest(TestCase):

    def test_environment(self):
        assert not(pyflamegpu.globalTelemetryEnabled())      # Should have been disabled globally
        assert ("FLAMEGPU_SILENCE_TELEMETRY_NOTICE" in os.environ)    # Should have been set globally
        assert ("FLAMEGPU_PYFLAMEGPU_VERSION" in os.environ) # Should have been set by module __init__