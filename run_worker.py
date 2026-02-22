"""
Custom worker runner that uses SimpleWorker (no forking)
This avoids fork() issues with HTTP connections and ML models
"""
from redis import Redis
from rq import Queue, SimpleWorker
import os
import logging

logging.basicConfig(level=logging.DEBUG)

redis_host = os.getenv('REDIS_HOST', 'localhost')
redis_port = int(os.getenv('REDIS_PORT', 6379))

conn = Redis(host=redis_host, port=redis_port)
queue = Queue(connection=conn)

if __name__ == '__main__':
    worker = SimpleWorker([queue], connection=conn)
    print("Starting SimpleWorker (no-fork mode)...", flush=True)
    worker.work(logging_level='DEBUG')
