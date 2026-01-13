import sys
import threading
from tqdm.auto import tqdm
from typing import Optional, Dict

_active_pbar = None
_worker_config = {} # Stores {'queue': multiprocessing.Queue, 'index': int}

def set_active_pbar(pbar):
    global _active_pbar
    _active_pbar = pbar

def get_active_pbar():
    return _active_pbar

def set_worker_config(queue, index):
    """Sets the IPC queue for progress reporting in a worker process."""
    global _worker_config
    _worker_config = {'queue': queue, 'index': index}

class MoeaProgress:
    """
    Manages progress bars for MoeaBench experiments.
    Supports both discrete steps (generations) and fractional progress (0-1).
    In worker processes, it redirects updates to a central queue.
    """
    def __init__(self, total=None, desc="MoeaBench", leave=True, position=0):
        self.total = total
        self.current_val = 0
        self.desc = desc
        self.queue = _worker_config.get('queue')
        self.worker_index = _worker_config.get('index')
        
        if self.queue:
            # Tell the manager we started a new run
            self.queue.put(('start', self.worker_index, desc, total))
            self.pbar = None
        else:
            # tqdm.auto automatically detects if we are in Jupyter or Terminal
            self.pbar = tqdm(total=100 if total is None else total, 
                             desc=desc, 
                             leave=leave, 
                             position=position,
                             unit="gen" if total is not None else "it",
                             ascii=" -")
        
        self.is_fractional = total is None

    def update_to(self, value):
        """
        Updates the progress bar to a specific value.
        """
        if self.is_fractional:
            # Map 0.0-1.0 to 0-100
            new_val = int(min(1.0, max(0.0, value)) * 100)
            diff = new_val - self.current_val
            if diff > 0:
                if self.queue:
                    self.queue.put(('update', self.worker_index, new_val))
                else:
                    self.pbar.update(diff)
                self.current_val = new_val
        else:
            val_int = int(value)
            diff = val_int - self.current_val
            if diff > 0:
                if self.queue:
                    self.queue.put(('update', self.worker_index, val_int))
                else:
                    self.pbar.update(diff)
                self.current_val = val_int

    def close(self):
        if self.queue:
            self.queue.put(('close', self.worker_index))
        else:
            # Ensure it hits 100% on close if it was close
            if self.is_fractional and self.current_val < 100:
                 self.pbar.update(100 - self.current_val)
            self.pbar.close()

    def set_description(self, desc):
        self.desc = desc
        if self.queue:
            self.queue.put(('desc', self.worker_index, desc))
        else:
            self.pbar.set_description(desc)

class ParallelProgressManager:
    """
    Coordinates multiple progress bars from different processes.
    """
    def __init__(self, repeat: int, desc: str, workers: int, queue):
        self.repeat = repeat
        self.desc = desc
        self.workers = workers
        self.queue = queue
        self.pbars: Dict[int, tqdm] = {}
        self.main_pbar = tqdm(total=repeat, desc=desc, position=0, ascii=" -")
        self.completed_runs = 0
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._listen, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        # Send a sentinel to unblock the queue if it's waiting
        # In this implementation, we rely on timeout or checking event
        self.thread.join(timeout=1.0)
        for pbar in self.pbars.values():
            pbar.close()
        self.main_pbar.close()

    def _listen(self):
        while not self.stop_event.is_set():
            try:
                # Use a timeout to allow checking stop_event
                msg = self.queue.get(timeout=0.1)
                cmd, idx = msg[0], msg[1]
                
                if cmd == 'start':
                    desc, total = msg[2], msg[3]
                    # Map idx to a position (idx is 1-based usually)
                    # We want workers to use positions 1 to workers
                    pos = (idx - 1) % self.workers + 1
                    if idx in self.pbars:
                        self.pbars[idx].close()
                    
                    self.pbars[idx] = tqdm(total=100 if total is None else total, 
                                          desc=desc, 
                                          position=pos, 
                                          leave=False,
                                          unit="gen" if total is not None else "it",
                                          ascii=" -")
                elif cmd == 'update':
                    val = msg[2]
                    if idx in self.pbars:
                        pbar = self.pbars[idx]
                        diff = val - pbar.n
                        if diff > 0:
                            pbar.update(diff)
                elif cmd == 'desc':
                    desc = msg[2]
                    if idx in self.pbars:
                        self.pbars[idx].set_description(desc)
                elif cmd == 'close':
                    if idx in self.pbars:
                        self.pbars[idx].close()
                        del self.pbars[idx]
                        self.completed_runs += 1
                        self.main_pbar.update(1)
            except Exception:
                continue

def get_progress_bar(total=None, desc="Optimizing", position=0, leave=True):
    """Factory to create a progress bar."""
    return MoeaProgress(total=total, desc=desc, position=position, leave=leave)
