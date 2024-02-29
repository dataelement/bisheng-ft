import pickle
from logging import getLogger

import redis
from redis import ConnectionPool
from settings import settings

logger = getLogger(__name__)


class RedisClient:

    def __init__(self, url, max_connections=10):
        self.pool = ConnectionPool.from_url(url,
                                            max_connections=max_connections)
        self.connection = redis.StrictRedis(connection_pool=self.pool)

    def set(self, key, value, expiration=3600):
        try:
            if pickled := pickle.dumps(value):
                result = self.connection.setex(key, expiration, pickled)
                if not result:
                    raise ValueError('RedisCache could not set the value.')
            else:
                logger.error('pickle error, value={}', value)
        except TypeError as e:
            raise TypeError(f'Redis accepts values that can be pickled {e}')
        finally:
            self.close()

    def set_no_expire(self, key, value):
        try:
            if pickled := pickle.dumps(value):
                result = self.connection.set(key, pickled)
                if not result:
                    raise ValueError('RedisCache could not set the value.')
            else:
                logger.error('pickle error, value={}', value)
        except TypeError as e:
            raise TypeError(f'Redis accepts values that can be pickled {e}')
        finally:
            self.close()

    def setNx(self, key, value, expiration=3600):
        try:
            if pickled := pickle.dumps(value):
                result = self.connection.setnx(key, pickled)
                self.connection.expire(key, expiration)
                if not result:
                    return False
                return True
        except TypeError as e:
            raise TypeError(f'Redis accepts values that can be pickled {e}')
        finally:
            self.close()

    def hsetkey(self, name, key, value, expiration=3600):
        try:
            r = self.connection.hset(name, key, value)
            if expiration:
                self.connection.expire(name, expiration)
            return r
        finally:
            self.close()

    def hset(self, name, map: dict, expiration=3600):
        try:
            r = self.connection.hset(name, mapping=map)
            if expiration:
                self.connection.expire(name, expiration)
            return r
        finally:
            self.close()

    def hget(self, name, key):
        try:
            return self.connection.hget(name, key)
        finally:
            self.close()

    def get(self, key):
        try:
            value = self.connection.get(key)
            return pickle.loads(value) if value else None
        finally:
            self.close()

    def delete(self, key):
        try:
            return self.connection.delete(key)
        finally:
            self.close()

    def exists(self, key):
        try:
            return self.connection.exists(key)
        finally:
            self.close()

    def close(self):
        self.connection.close()

    def __contains__(self, key):
        """Check if the key is in the cache."""
        return False if key is None else self.connection.exists(key)

    def __getitem__(self, key):
        """Retrieve an item from the cache using the square bracket notation"""
        return self.get(key)

    def __setitem__(self, key, value):
        """Add an item to the cache using the square bracket notation."""
        self.set(key, value)

    def __delitem__(self, key):
        """Remove an item from the cache using the square bracket notation."""
        self.delete(key)


# 示例用法
redis_client = RedisClient(settings.redis_url)
