import pytest
from unittest import TestCase
from pyflamegpu import *

class JitifyCacheTest(TestCase):
    """
        Test the now exposed flamegpu::detail::JitifyCache methods are exposed to python
        This does not test the clear*cache methods, as that would have grim side-effects
    """
    def test_memorycache(self):
        """
        Test setting and checking the state of the jitify memory cache
        """
        rtc_cache = pyflamegpu.JitifyCache.getInstance()
        originally_enabled = rtc_cache.useMemoryCache()
        rtc_cache.useMemoryCache(True)
        assert rtc_cache.useMemoryCache() == True
        rtc_cache.useMemoryCache(False)
        assert rtc_cache.useMemoryCache() == False
        rtc_cache.useMemoryCache(originally_enabled)
        assert rtc_cache.useMemoryCache() == originally_enabled

    def test_diskcache(self):
        """
        Test setting and checking the state of the jitify disk cache
        """
        rtc_cache = pyflamegpu.JitifyCache.getInstance()
        originally_enabled = rtc_cache.useDiskCache()
        rtc_cache.useDiskCache(True)
        assert rtc_cache.useDiskCache() == True
        rtc_cache.useDiskCache(False)
        assert rtc_cache.useDiskCache() == False
        rtc_cache.useDiskCache(originally_enabled)
        assert rtc_cache.useDiskCache() == originally_enabled
