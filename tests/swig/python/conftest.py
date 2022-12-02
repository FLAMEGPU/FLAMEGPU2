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
    session.telemetry  = pyflamegpu.globalTelemetryEnabled()
    # Disable any global telemetry to prevent telemetry for every simulation in test suite
    os.environ['FLAMEGPU_SHARE_USAGE_STATISTICS'] = 'False'
    # Silence telemetry notice warning in test suite
    os.environ['FLAMEGPU_SILENCE_TELEMETRY_NOTICE'] = 'True'


def pytest_sessionfinish(session, exitstatus):
    """
    Called after whole test run finished, right before returning the exit status to the system.
    This hook is used to send a single telemetry packed for the test if enabled in cmake or the 
    environment variable.
    """
    # ensure session has telemetry value set by pytest_sessionstart
    if not hasattr(session, "telemetry"):
        print("'telemetry' value not found in session.")
        return
    # if telemetry was enabled prior to testing then send telemetry data
    if session.telemetry:
        telemetry_payload =  {}
        # store outcome in payload
        if exitstatus in [0, 5]: # pass, no tests collected
            telemetry_payload["TestOutcome"] = "Passed"
        else:
            telemetry_payload["TestOutcome"] = f"Failed(code={exitstatus})"
        # get the terminal reporter to query pass and fails
        terminalreporter = session.config.pluginmanager.get_plugin('terminalreporter')
        # store test details in payload
        telemetry_payload['TestsCollected'] = str(len(session.items))
        telemetry_payload['TestsPassed'] = str(len(terminalreporter.stats.get('passed', [])))
        telemetry_payload['TestsFailed']=  str(len(terminalreporter.stats.get('failed', [])))
        telemetry_payload['TestsSkipped']=  str(len(terminalreporter.stats.get('skipped', [])))
        telemetry_payload['TestsDeselected']=  str(len(terminalreporter.stats.get('deselected', [])))
        # generate telemetry data (convert dict to map<string, string>)
        telemetry_data = pyflamegpu.generateTelemetryData("pytest-run", pyflamegpu.map_string_string(telemetry_payload))
        # send telemetry
        pyflamegpu.sendTelemetryData(telemetry_data)
        # print
        if session.config.getoption("verbose") > 0:
            print(f"\nTelemetry packet sent to '{pyflamegpu.TELEMETRY_ENDPOINT}' json was: {telemetry_data}\n")
