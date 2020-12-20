import numpy as np
import torch
import torch.nn as nn
import warnings
from datetime import timedelta, datetime
from types import SimpleNamespace
from typing import Iterable, Tuple, List

import ptan
import ptan.ignite as ptan_ignite
from ignite.engine import Engine
from ignite.metrics import RunningAverage
from ignite.contrib.handlers import tensorboard_logger as tb_logger

# Seed, so that results can be compared
SEED = 123

HYPERPARAMS = {
    'pong': SimpleNamespace(**{
        'env_name':         "PongNoFrameskip-v4",
        'stop_reward':      18.0,
        'run_name':         'pong',
        'replay_size':      100000,
        'replay_initial':   10000,
        'target_net_sync':  1000,
        'epsilon_frames':   10**5,
        'epsilon_start':    1.0,
        'epsilon_final':    0.02,
        'learning_rate':    0.0001,
        'gamma':            0.99,
        'batch_size':       32
    }),
    'breakout-small': SimpleNamespace(**{
        'env_name':         "BreakoutNoFrameskip-v4",
        'stop_reward':      500.0,
        'run_name':         'breakout-small',
        'replay_size':      3*10 ** 5,
        'replay_initial':   20000,
        'target_net_sync':  1000,
        'epsilon_frames':   10 ** 6,
        'epsilon_start':    1.0,
        'epsilon_final':    0.1,
        'learning_rate':    0.0001,
        'gamma':            0.99,
        'batch_size':       64
    }),
    'breakout': SimpleNamespace(**{
        'env_name':         "BreakoutNoFrameskip-v4",
        'stop_reward':      500.0,
        'run_name':         'breakout',
        'replay_size':      10 ** 6,
        'replay_initial':   50000,
        'target_net_sync':  10000,
        'epsilon_frames':   10 ** 6,
        'epsilon_start':    1.0,
        'epsilon_final':    0.1,
        'learning_rate':    0.00025,
        'gamma':            0.99,
        'batch_size':       32
    }),
    'invaders': SimpleNamespace(**{
        'env_name': "SpaceInvadersNoFrameskip-v4",
        'stop_reward': 500.0,
        'run_name': 'breakout',
        'replay_size': 10 ** 6,
        'replay_initial': 50000,
        'target_net_sync': 10000,
        'epsilon_frames': 10 ** 6,
        'epsilon_start': 1.0,
        'epsilon_final': 0.1,
        'learning_rate': 0.00025,
        'gamma': 0.99,
        'batch_size': 32
    }),
}

# A function that takes batch of transitions and converts it into Numpy. This works from ExperienceSOurceFirstLast only
# Every transition is from ExperienceSourceFirstLast, whose output is a tuple
# state(obs from env), action(taken by agent), reward(either immediate or discounted depends on initialization) and 
# last_state -> if transition corresponds to final step in env, then None, else the last obs from env
def unpack_batch(batch: List[ptan.experience.ExperienceFirstLast]):
    states, actions, rewards, dones, last_states = [],[],[],[],[]
    for exp in batch:
        state = np.array(exp.state)
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None) # A list of True/False 
        if exp.last_state is None:
            lstate = state  # The result will be masked anyway
        else:
            lstate = np.array(exp.last_state)
        last_states.append(lstate)
        # Note how last_state is not appended when last_state is done. 
        # This is to avoid special handling (Bellman update) of such cases 
    return np.array(states, copy=False), np.array(actions), \
           np.array(rewards, dtype=np.float32), \
           np.array(dones, dtype=np.uint8), \
           np.array(last_states, copy=False)

# Calculation of loss of dqn. For more info, check the debugging module (a jupyter notebook)
def calc_loss_dqn(batch, net, tgt_net, gamma, device="cpu"):
    # Unpack the states, actions, rewards, done_bools and next-states. Convert them all into tensors. 
    states, actions, rewards, dones, next_states = unpack_batch(batch)

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    # Get the state-action value(s) [values as batch size is generally > 1]
    actions_v = actions_v.unsqueeze(-1) 
    state_action_vals = net(states_v).gather(1, actions_v) # The 1 is for dimension-1
    state_action_vals = state_action_vals.squeeze(-1)
    
    with torch.no_grad():
        next_state_vals = tgt_net(next_states_v).max(1)[0]
        next_state_vals[done_mask] = 0.0

    # Bellman equation
    bellman_vals = next_state_vals.detach() * gamma + rewards_v

    return nn.MSELoss()(state_action_vals, bellman_vals)

# A class to implement epsilon decay
class EpsilonTracker:
    def __init__(self, selector: ptan.actions.EpsilonGreedyActionSelector, params: SimpleNamespace):
        self.selector = selector
        self.params = params
        self.frame(0)

    def frame(self, frame_idx: int):
        eps = self.params.epsilon_start - frame_idx / self.params.epsilon_frames
        self.selector.epsilon = max(self.params.epsilon_final, eps)

# A function that infinitely generates training batches sampled from buffer
def batch_generator(buffer: ptan.experience.ExperienceReplayBuffer, initial: int, batch_size: int):
    buffer.populate(initial)
    while True:
        buffer.populate(1)
        yield buffer.sample(batch_size)


@torch.no_grad()
def calc_values_of_states(states, net, device="cpu"):
    mean_vals = []
    for batch in np.array_split(states, 64):
        states_v = torch.tensor(batch).to(device)
        action_values_v = net(states_v)
        best_action_values_v = action_values_v.max(1)[0]
        mean_vals.append(best_action_values_v.mean().item())
    return np.mean(mean_vals)

# A function that shows training progress and writing metrics to Tensorboard
def setup_ignite(engine: Engine, params: SimpleNamespace, exp_source, run_name: str, extra_metrics: Iterable[str] = ()):

    # get rid of missing metrics warning
    warnings.simplefilter("ignore", category=UserWarning)

    # EndofEpisodeHandler emits Ignite event every time a game episode ends
    handler = ptan_ignite.EndOfEpisodeHandler(exp_source, bound_avg_reward=params.stop_reward)
    handler.attach(engine)

    # EpisodeFPSHandler tracks time the episode has taken and amount of interactions
    ptan_ignite.EpisodeFPSHandler().attach(engine)

    # An event handler to call at end of episode
    @engine.on(ptan_ignite.EpisodeEvents.EPISODE_COMPLETED)
    def episode_completed(trainer: Engine):
        passed = trainer.state.metrics.get('time_passed', 0)
        print("Episode {}: reward={:.0f}, steps={}, " "speed={:.1f} f/s, elapsed={}".format(trainer.state.episode, trainer.state.episode_reward, trainer.state.episode_steps, trainer.state.metrics.get('avg_fps', 0), timedelta(seconds=int(passed))))

    # An event handler to call when avg reward grows above boundary
    @engine.on(ptan_ignite.EpisodeEvents.BOUND_REWARD_REACHED)
    def game_solved(trainer: Engine):
        passed = trainer.state.metrics['time_passed']
        print("Game solved in {}, after {} episodes and {} iterations!".format(timedelta(seconds=int(passed)), trainer.state.episode, trainer.state.iteration))
        trainer.should_terminate = True

    # Initialize few variables to create the tensorboardlogger directory
    now = datetime.now().isoformat(timespec='minutes')
    logdir = f"runs/{now}-{params.run_name}-{run_name}"
    
    # Create a TensorboardLogger to write into Tensorboard
    tb = tb_logger.TensorboardLogger(log_dir=logdir)

    # Obtain the running average transoformation to get a smoothed version of loss over time.
    run_avg = RunningAverage(output_transform=lambda v: v['loss'])
    run_avg.attach(engine, "avg_loss")

    # TensorboadLogger can track two groups of values from Ignite: outputs(values returned by transformation function) and metrics
    metrics = ['reward', 'steps', 'avg_reward']
    handler = tb_logger.OutputHandler(tag="episodes", metric_names=metrics)

    event = ptan_ignite.EpisodeEvents.EPISODE_COMPLETED # Write after every episode
    tb.attach(engine, log_handler=handler, event_name=event)

    # Write to tensorboard every 100 iterations, tracking the metrics from training process
    ptan_ignite.PeriodicEvents().attach(engine)
    metrics = ['avg_loss', 'avg_fps']
    metrics.extend(extra_metrics)
    handler = tb_logger.OutputHandler(tag="train", metric_names=metrics, output_transform=lambda a: a)
    event = ptan_ignite.PeriodEvents.ITERS_100_COMPLETED
    tb.attach(engine, log_handler=handler, event_name=event)
