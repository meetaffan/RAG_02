import os
from redis import Redis
from rq import Queue

# Use environment variable for Redis host, default to localhost for local development
# Git push
redis_host = os.getenv('REDIS_HOST', 'localhost')
redis_port = int(os.getenv('REDIS_PORT', 6379))

q = Queue(connection=Redis(host=redis_host, port=redis_port))

