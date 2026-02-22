"""
Custom RQ Worker that doesn't fork
Runs jobs in the same process to avoid fork() issues with ML models
"""
from rq.worker import Worker

class NoForkWorker(Worker):
    """
    Worker that executes jobs in the same process without forking
    This avoids issues with ML models and GPU contexts that don't fork well
    """
    def execute_job(self, job, queue):
        """Execute job in the same process (no fork)"""
        # Set up job execution context
        self.prepare_job_execution(job)
        
        # Run the job directly in this process
        with self.connection.pipeline() as pipeline:
            try:
                job.started_at = job.ended_at = None
                job.set_status('started', pipeline=pipeline)
                pipeline.execute()
                
                # Execute the actual job function
                rv = job.perform()
                
                # Job succeeded
                job.ended_at = job.utcnow()
                job._result = rv
                
                if rv is not None:
                    job.save(pipeline=pipeline, include_meta=False)
                
                job.set_status('finished', pipeline=pipeline)
                job.cleanup(ttl=job.result_ttl, pipeline=pipeline)
                pipeline.execute()
                
                return True
                
            except Exception:
                # Job failed
                job.handle_failure(exc_info=True)
                return False
