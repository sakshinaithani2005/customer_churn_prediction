import time
import unittest
from proxy.gateway import RateLimiter


class TestRateLimiter(unittest.TestCase):
    def test_rate_limiter_allows_under_limit(self):
        rl = RateLimiter(max_requests=3, window_seconds=1.0)
        self.assertFalse(rl.is_rate_limited("127.0.0.1"))
        self.assertFalse(rl.is_rate_limited("127.0.0.1"))
        self.assertFalse(rl.is_rate_limited("127.0.0.1"))

    def test_rate_limiter_blocks_over_limit(self):
        rl = RateLimiter(max_requests=2, window_seconds=1.0)
        self.assertFalse(rl.is_rate_limited("192.168.1.1"))
        self.assertFalse(rl.is_rate_limited("192.168.1.1"))
        self.assertTrue(rl.is_rate_limited("192.168.1.1"))

    def test_rate_limiter_resets_after_window(self):
        rl = RateLimiter(max_requests=2, window_seconds=0.2)
        self.assertFalse(rl.is_rate_limited("10.0.0.1"))
        self.assertFalse(rl.is_rate_limited("10.0.0.1"))
        self.assertTrue(rl.is_rate_limited("10.0.0.1"))

        time.sleep(0.25)
        self.assertFalse(rl.is_rate_limited("10.0.0.1"))


if __name__ == "__main__":
    unittest.main()
