import pytest
import os
import sys
import pyflamegpu

def pytest_sessionstart(session):
    """
    Hook to execute environmental setup prior to executing tests. The Telemetry environment 
    variable is stored in the session and then set to 'False' to prevent telemetry spam as 
    a result of running tests.
    """
    # Get the global telemetry value (either set in cmake or env) and store in session for later
    session.telemetry  = pyflamegpu.Telemetry.isEnabled()
    pyflamegpu.Telemetry.disable()
    pyflamegpu.Telemetry.suppressNotice()
