import time

def measure_execution_time(func, *args, **kwargs):
    """
    Measures the execution time of a function while excluding time spent waiting for user input.

    Args:
        func: The function to measure
        *args, **kwargs: Arguments to pass to the function

    Returns:
        tuple: (function result, execution time in seconds)
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    execution_time = end_time - start_time
    if result is not None:
        return result, end_time - start_time
    else:
        return execution_time

def timed_input(prompt=""):
    """
    Gets user input while excluding the waiting time from execution measurements.

    Args:
        prompt: Input prompt to display

    Returns:
        tuple: (user input, time spent waiting in seconds)
    """
    pause_time = time.time()
    user_input = input(prompt)
    resume_time = time.time()
    return user_input, resume_time - pause_time

class ExecutionTimer:
    """
    Class for tracking execution time across multiple code segments,
    excluding time spent waiting for user input.
    """
    def __init__(self):
        self.total_time = 0
        self.wait_time = 0
        self.start_time = None
        self.is_running = False

    def start(self):
        """Start the timer"""
        if not self.is_running:
            self.start_time = time.time()
            self.is_running = True

    def stop(self):
        """Stop the timer and return the elapsed time"""
        if self.is_running:
            self.total_time += time.time() - self.start_time
            self.is_running = False
            return self.total_time
        return None

    def add_wait_time(self, wait_time):
        """Add user input waiting time to be excluded from total execution time"""
        self.wait_time += wait_time

    def get_active_time(self):
        """Return the active execution time (total time - wait time)"""
        return self.total_time - self.wait_time

    def reset(self):
        """Reset the timer"""
        self.total_time = 0
        self.wait_time = 0
        self.start_time = None
        self.is_running = False