from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy


def run_task(*_):
    env = TfEnv(normalize(SwimmerEnv()))
    policy = GaussianMLPPolicy(
        name="policy",
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(32, 32)
    )
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=4000,
        max_path_length=500,
        n_itr=40,
        discount=0.99,
        step_size=0.01,
        plot=True,
    )
    algo.train()


run_experiment_lite(
    run_task,
    n_parallel=4,
    snapshot_mode="last",
    seed=1,
    # Please uncomment this line and replace this to your own python environment if needed
    # python_command='/home/xxxx/.conda/envs/rllab3/bin/python',
    plot=True,
    backend='Tensorflow',
)