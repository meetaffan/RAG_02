import uvicorn
from server import app
from .queue.connection import connection
from .queue.worker import process_query

def main():
    uvicorn.run(app, port = 8000, host = "0.0.0.0")

main()
