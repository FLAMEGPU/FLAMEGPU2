import pytest
import os
import sys
import pyflamegpu

"""
Use pytest a pytest class fixture and a pytest sessionfinish hooks to handle telemetry

The class fixture is executed per test class for all test classes within this directory or below. 
It records the telemetry enabled state, disables telemetry, and then restores telemetry to the original value.  

If telemetry is enabled and more than one test was executed, submit the test results to telemetry deck at session end. 

We cannot rely on sessionstart incase the pytest entry point is above this file, so disabling and restoring telemetry per test class is the more reliable option.
"""

@pytest.fixture(scope="class", autouse=True)
def class_fixture():
    """Class scoped fixture to disable telemetry, ensuring this is done for all tests below this conftest.py file, regardless of whether the pytest entry point was above this (i.e. it isn't reliable to do in a session_start.)
    """
    # Get the current value
    was_enabled = pyflamegpu.Telemetry.isEnabled()
    # Disable telemetry
    pyflamegpu.Telemetry.disable()
    # Disable the suppression notice
    pyflamegpu.Telemetry.suppressNotice()
    yield
    # Set telemetry back to the original value, this avoids the need for the unreliable session_start call. 
    if was_enabled:
        pyflamegpu.Telemetry.enable()

def pytest_sessionfinish(session, exitstatus):
    """Hook to execute code during session tear down, once all tests have been executed, and the final status is known. 
    If telemetry is enabled (fixture re-enables if required) submit test result telemetry as long as more than one test was executed (to avoid 3rd party tool test running spamming the API).
    """
    # only submit telemetry if it was originally enabled
    if pyflamegpu.Telemetry.isEnabled():
        # get the terminal reporter to query pass and fails
        terminalreporter = session.config.pluginmanager.get_plugin('terminalreporter')
        # Exit if the terminalreport plugin could not be found
        if not terminalreporter:
            return
        outcome = "Passed" if exitstatus == 0 else "Failed(code={exitstatus})"
        passed = len(terminalreporter.stats.get('passed', []))
        failed = len(terminalreporter.stats.get('failed', []))
        skipped = len(terminalreporter.stats.get('skipped', []))
        deselected = len(terminalreporter.stats.get('deselected', []))
        total = passed + failed + skipped + deselected
        selected = passed + failed

        # If telemetry was enabled, and more than 1 test was executed
        if selected > 1:
            # Send the results to telemetry deck, using the wrapped but privatised method, silently fail if the curl request fails.
            pyflamegpu._pyflamegpu.__TestSuiteTelemetry_sendResults("pytest-run"
                , outcome
                , total
                , selected
                , skipped
                , passed
                , failed
                , session.config.getoption("verbose") > 0
                , True) # True this was from Python