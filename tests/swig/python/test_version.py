import pytest
from unittest import TestCase
from pyflamegpu import *
import sys


if sys.version_info >= (3, 8):
    from importlib.metadata import version

class VersionTest(TestCase):

    def test_version(self):
        assert pyflamegpu.VERSION > 0
        assert pyflamegpu.VERSION_MAJOR >= 2
        assert pyflamegpu.VERSION_MINOR >= 0
        assert pyflamegpu.VERSION_PATCH >= 0
        combined_version_parts = (pyflamegpu.VERSION_MAJOR * 1000000) + (pyflamegpu.VERSION_MINOR * 1000) + (pyflamegpu.VERSION_PATCH)
        assert pyflamegpu.VERSION == combined_version_parts
        assert type(pyflamegpu.VERSION_PRERELEASE) == str
        assert type(pyflamegpu.VERSION_BUILDMETADATA) == str
        assert type(pyflamegpu.VERSION_STRING) == str
        assert type(pyflamegpu.VERSION_FULL) == str

        module_prerelease = ""
        split_prerelease = pyflamegpu.VERSION_PRERELEASE.split(".")
        if len(split_prerelease) > 0:
            if split_prerelease[0] == "alpha":
                module_prerelease = "a"
            elif split_prerelease[0] == "beta":
                module_prerelease = "b"
            elif split_prerelease[0] == "rc":
                module_prerelease = "rc"
            else:
                assert False
            if len(split_prerelease) > 1:
                prerelease_num = split_prerelease[1]
                module_prerelease += prerelease_num
            else:
                module_prerelease += "0"
        expected_module_version = f"{pyflamegpu.VERSION_MAJOR}.{pyflamegpu.VERSION_MINOR}.{pyflamegpu.VERSION_PATCH}{module_prerelease}"
        if sys.version_info >= (3, 8):
            # The module version may include a local version string (+), so we only want to compare up to the first +.
            split_pub_local_version = version('pyflamegpu').split("+")
            public_version = split_pub_local_version[0]
            local_version = split_pub_local_version[1] if len(split_pub_local_version) > 1 else ""
            assert expected_module_version == public_version