import atexit
import time
from queue import Empty, Queue
from threading import Thread

import numpy as np
import tensorflow as tf

from rllab.sampler.utils import rollout


__all__ = [
    'init_worker',
    'init_plot',
    'update_plot',
]

plotter_thread = None
queue = None


def _worker_start(sess):
    env = None
    policy = None
    max_length = None
    while True:
        try:
            time.sleep(1e-6)
            msgs = {}
            # Only fetch the last message of each type
            with sess.as_default():
                while True:
                    try:
                        msg = queue.get_nowait()
                        msgs[msg[0]] = msg[1:]
                    except Empty:
                        break
                if 'stop' in msgs:
                    break
                elif 'update' in msgs:
                    env, policy = msgs['update']
                elif 'demo' in msgs:
                    param_values, max_length = msgs['demo']
                    policy.set_param_values(param_values)
                    rollout(
                        env,
                        policy,
                        max_path_length=max_length,
                        animated=True,
                        speedup=5)
                else:
                    if max_length:
                        rollout(
                            env,
                            policy,
                            max_path_length=max_length,
                            animated=True,
                            speedup=5)
        except KeyboardInterrupt:
            break
        except RuntimeError as e:
            if e == 'Attempted to use a closed Session.':
                break


def _shutdown_worker():
    if plotter_thread:
        queue.put_nowait(['stop'])
        plotter_thread.join()


def init_worker(sess=None):
    global plotter_thread, queue
    queue = Queue()
    if sess is None:
        sess = tf.get_default_session()
    plotter_thread = Thread(target=_worker_start, args=(sess,))
    plotter_thread.setDaemon(True)
    plotter_thread.start()
    atexit.register(_shutdown_worker)


def init_plot(env, policy):
    queue.put_nowait(['update', env, policy])


def update_plot(policy, max_length=np.inf):
    queue.put_nowait(['demo', policy.get_param_values(), max_length])
