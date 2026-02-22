"""
Custom RQ Worker that doesn't fork
Runs jobs in the same process to avoid fork() issues with ML models
"""
from rq.worker import Worker
import signal

class NoForkWorker(Worker):
    """
    Worker that executes jobs in the same process without forking
    This avoids issues with ML models and GPU contexts that don't fork well
    """
    
    def main_work_horse(self, *args, **kwargs):
        """
        Override main_work_horse to prevent forking
        Returns None instead of PID to signal in-process execution
        """
        # Signal that we're not forking (execute in same process)
        return None
        
    def handle_job_failure(self, job, exc_info=None):
        """Handle job failure without forking"""
        with self.connection.pipeline() as pipeline:
            job.set_status('failed', pipeline=pipeline)
            job.save(pipeline=pipeline)
            pipeline.execute()
            
    def handle_job_success(self, job, queue, started_job_registry):
        """Handle job success without forking"""
        with self.connection.pipeline() as pipeline:
            job.set_status('finished', pipeline=pipeline)
            job.save(pipeline=pipeline)
            started_job_registry.remove(job, pipeline=pipeline)
            pipeline.execute()

