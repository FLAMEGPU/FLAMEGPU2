import pytest
from unittest import TestCase
from pyflamegpu import *
import os

class TelemetryTest(TestCase):
    def test_TelemetryEnableDisable(self):
        # Telemetry has been disabled by the test suite handle, so it should be off by default at this point in time. We cannot check the initial parsing of this value unfortunately (without making a copy in a global?)
        assert pyflamegpu.Telemetry.isEnabled() == False
        # Turn telemetry on, check it is enabled.
        pyflamegpu.Telemetry.enable()
        assert pyflamegpu.Telemetry.isEnabled() == True
        #  Turn it off again, making sure it is disabled
        pyflamegpu.Telemetry.disable()
        assert pyflamegpu.Telemetry.isEnabled() == False


    def test_TelemetrySuppressNotice(self):
        # We cannot check the value of suppression, and cannot re-enable the suppression warning with the protection / annon namespace, so all we can do is check the method exists.
        pyflamegpu.Telemetry.suppressNotice()