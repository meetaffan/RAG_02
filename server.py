from fastapi import FastAPI, Query, HTTPException
from task_queue.connection import q
from task_queue.worker import process_query
from rq.job import Job
from rq.registry import StartedJobRegistry, FinishedJobRegistry, FailedJobRegistry

app = FastAPI()

@app.get('/')
def home():
    return {"status": 'Server is up and running'}

@app.post('/chat')
def chat(query: str = Query(..., description="Chat Message")):
    job = q.enqueue(process_query, query)
    return {"status": "queued", "job_id": job.id}

@app.get('/result/{job_id}')
def get_result(job_id: str):
    """Get the result of a queued job"""
    try:
        job = Job.fetch(job_id, connection=q.connection)
        
        if job.is_finished:
            return {
                "status": "completed",
                "result": job.result
            }
        elif job.is_failed:
            return {
                "status": "failed",
                "error": str(job.exc_info)
            }
        elif job.is_started:
            return {
                "status": "processing",
                "message": "Job is currently being processed"
            }
        else:
            return {
                "status": "queued",
                "message": "Job is waiting in queue"
            }
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Job not found: {str(e)}")

@app.get('/queue/status')
def queue_status():
    """Get the current queue status"""
    started_registry = StartedJobRegistry(queue=q)
    finished_registry = FinishedJobRegistry(queue=q)
    failed_registry = FailedJobRegistry(queue=q)
    
    # Get queued jobs
    queued_jobs = []
    for job in q.jobs:
        queued_jobs.append({
            "job_id": job.id,
            "created_at": str(job.created_at),
            "args": job.args
        })
    
    # Get processing jobs
    processing_jobs = []
    for job_id in started_registry.get_job_ids():
        try:
            job = Job.fetch(job_id, connection=q.connection)
            processing_jobs.append({
                "job_id": job.id,
                "started_at": str(job.started_at),
                "args": job.args
            })
        except:
            pass
    
    return {
        "queued": len(q),
        "processing": len(started_registry),
        "completed": len(finished_registry),
        "failed": len(failed_registry),
        "queued_jobs": queued_jobs,
        "processing_jobs": processing_jobs
    }

