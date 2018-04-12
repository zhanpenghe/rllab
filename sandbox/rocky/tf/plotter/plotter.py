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


def _worker_start():
    env = None
    policy = None
    sess = None
    max_length = None
    while True:
        try:
            time.sleep(1e-6)
            msgs = {}
            # Only fetch the last message of each type
            while True:
                try:
                    msg = queue.get_nowait()
                    msgs[msg[0]] = msg[1:]
                except Empty:
                    break
            if 'stop' in msgs:
                sess.close()
                break
            elif 'update' in msgs:
                env, policy, sess = msgs['update']
            elif 'demo' in msgs:
                with sess.as_default():
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
                    with sess.as_default():
                        rollout(
                            env,
                            policy,
                            max_path_length=max_length,
                            animated=True,
                            speedup=5)
        except KeyboardInterrupt:
            break
        except RuntimeError as e:
            # Handle pygame environment lifecycle problems..
            if e == 'video system not initialized':
                break


def _shutdown_worker():
    if plotter_thread:
        queue.put_nowait(['stop'])
        plotter_thread.join()


def init_worker():
    global plotter_thread, queue
    queue = Queue()
    plotter_thread = Thread(target=_worker_start)
    plotter_thread.setDaemon(True)
    plotter_thread.start()
    atexit.register(_shutdown_worker)


def init_plot(env, policy, graph=None):
    if graph is None:
        graph = tf.get_default_graph()

    sess = tf.Session(graph=graph)
    sess.run(tf.global_variables_initializer())
    queue.put_nowait(['update', env, policy, sess])


def update_plot(policy, max_length=np.inf):
    queue.put_nowait(['demo', policy.get_param_values(), max_length])
