from fastapi import FastAPI, Query
from task_queue.connection import q
from task_queue.worker import process_query

app = FastAPI()

@app.get('/')
def chat():
    return {"status": 'Server is up and running'}

@app.post('/chat')
def chat( query: str = Query(..., description ="Chat Message") ):
    job = q.enqueue(process_query, query)

    return {"status" : "queued", "job_id" : job.id}

