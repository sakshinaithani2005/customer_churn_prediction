import time
import unittest
from proxy.gateway import LruCache


class TestLruCache(unittest.TestCase):
    def test_cache_put_and_get(self):
        cache = LruCache(capacity=2, ttl_seconds=10.0)
        key1 = LruCache.build_key(b'{"CreditScore": 600}', "v1")
        key2 = LruCache.build_key(b'{"CreditScore": 700}', "v1")

        cache.put(key1, '{"churn": 0}', status_code=200)
        cache.put(key2, '{"churn": 1}', status_code=200)

        entry1 = cache.get(key1)
        self.assertIsNotNone(entry1)
        self.assertEqual(entry1.body, '{"churn": 0}')

        entry2 = cache.get(key2)
        self.assertIsNotNone(entry2)
        self.assertEqual(entry2.body, '{"churn": 1}')

    def test_cache_lru_eviction(self):
        cache = LruCache(capacity=2, ttl_seconds=10.0)
        key1 = "k1"
        key2 = "k2"
        key3 = "k3"

        cache.put(key1, "val1")
        cache.put(key2, "val2")

        # Touch k1 so k2 becomes LRU
        _ = cache.get(key1)

        # Insert k3, should evict k2
        cache.put(key3, "val3")

        self.assertIsNotNone(cache.get(key1))
        self.assertIsNone(cache.get(key2))
        self.assertIsNotNone(cache.get(key3))

    def test_cache_ttl_expiration(self):
        cache = LruCache(capacity=2, ttl_seconds=0.1)
        key = "short-lived"
        cache.put(key, "data")
        self.assertIsNotNone(cache.get(key))

        time.sleep(0.15)
        self.assertIsNone(cache.get(key))


if __name__ == "__main__":
    unittest.main()
