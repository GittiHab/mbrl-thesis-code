import abc
from copy import deepcopy
from typing import Optional

from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from models.models import bottle
import torch
from stable_baselines3.common.torch_layers import MlpExtractor
from exploration.base import Explorer
from utils import reset_model_params
from .optimizer import Optimizer
from .targets import one_step_td, get_target_function
from warnings import warn
from torch.optim.lr_scheduler import CyclicLR, StepLR

__all__ = ['RewardSpace', 'ExplorationRewardSpace', 'setup_reward_space']


def setup_reward_space(config, world_model, lr_step_size, name):
    if config.value_lr is not None:
        reward_space = RewardSpace(world_model.reward_model,
                                   value_lr=config.value_lr,
                                   value_layers=config.value_layers if 'value_layers' in config else 3,
                                   state_size=world_model.state_size,
                                   hidden_size=config.hidden_size if 'hidden_size' in config else world_model.hidden_size,
                                   lr_strategy=setup_lr_schedule(config, name, lr_step_size),
                                   device=world_model.device,
                                   name=name,
                                   value_target_fn=get_target_function(config),
                                   update_target_every=config.update_target_every if 'update_target_every' in config else None)
    else:
        reward_space = RewardSpace(world_model.reward_model)
    return reward_space


def setup_lr_schedule(config, name, lr_step_size):
    strategy = 'step' if 'lr_strategy' not in config else config.lr_strategy.lower()
    if strategy == 'step':
        if lr_step_size <= 0:
            warn('Using no LR scheduler as step size is smaller than 1.')
            return None
        if 'lr_decay' not in config:
            return None
        min_lr = config.value_lr * config.lr_min_factor if 'lr_decay' in config else config.value_lr * 0.01
        return StepLRSchedule(name, lr_step_size, config.lr_decay, min_lr)
    elif strategy == 'cyclic':
        return CyclicLRSchedule(name, min_lr=config.lr_min, max_lr=config.lr_max,
                                lr_half_cycle=config.lr_half_cycle, lr_mode=config.lr_mode)
    warn('Specified learning rate schedule not supported.')
    return None


class LearningRateStrategy(abc.ABC):

    def __init__(self, name, min_lr=None):
        self.name = name
        self.scheduler = None
        self.min_lr = min_lr
        self._last_lr = None

    def _step(self, logger):
        logger.record("train/learning_rate_" + self.name, self.last_lr)
        if self.min_lr is None or self.last_lr > self.min_lr:
            self.scheduler.step()
            self._last_lr = self.scheduler.get_last_lr()[0]

    @property
    def last_lr(self):
        if self._last_lr is None:
            return self.scheduler.get_last_lr()[0]
        return self._last_lr

    def after_batch(self, logger):
        pass

    def after_epoch(self, logger):
        pass

    @abc.abstractmethod
    def reset_schedule(self, optimizer):
        pass


class CyclicLRSchedule(LearningRateStrategy):

    def __init__(self, name, min_lr, max_lr, lr_half_cycle, lr_mode):
        super().__init__(name)
        self.lr_min = min_lr
        self.lr_max = max_lr
        self.lr_half_cycle = lr_half_cycle
        self.lr_mode = lr_mode

        self.scheduler = None  # type: CyclicLR

    def after_batch(self, logger):
        self._step(logger)

    def reset_schedule(self, optimizer):
        self.scheduler = CyclicLR(optimizer, base_lr=self.lr_min, max_lr=self.lr_max,
                                  step_size_up=self.lr_half_cycle, mode=self.lr_mode, cycle_momentum=False)


class StepLRSchedule(LearningRateStrategy):
    def __init__(self, name, lr_step_size, lr_decay, min_lr=None):
        super().__init__(name, min_lr)
        self.lr_step_size = lr_step_size
        self.lr_decay = lr_decay

        self.scheduler = None  # type: StepLR

    def after_epoch(self, logger):
        self._step(logger)

    def reset_schedule(self, optimizer):
        self.scheduler = StepLR(optimizer, self.lr_step_size, self.lr_decay)
        return self.scheduler


class RewardSpace(Optimizer):
    def __init__(self, reward_model, value_lr=None, hidden_size=None, state_size=None, belief_size=None, device=None,
                 value_layers=3, lr_strategy: Optional[LearningRateStrategy] = None, grad_clip_norm=100.0,
                 name='', value_target_fn=one_step_td, update_target_every=1):
        '''

        :param reward_model: Model that predicts the reward given belief, state, and action.
        :param value_lr: Learning rate of the value network. Leave None to NOT use a value network.
        :param hidden_size: Hidden layer size of the value network. Required when value network is used.
        :param state_size: State size, if None only the belief is used as an input for the value network.
        :param belief_size: Belief size, if None only the state is used as an input for the value network.
                            belief_size and state_size cannot both be None if value network is used.
        :param device: PyTorch device (cuda or cpu).
        :param value_layers: Number of layers of the value network.
        :param name: Name of this reward space. Will be shown in the log.
        '''
        self.reward_model = reward_model
        self._name = name

        self.lr_strategy = lr_strategy
        if value_lr is not None:
            assert hidden_size is not None and (
                    state_size is not None or belief_size is not None) and device is not None, \
                'If value network is used, value_lr, hidden_size, device, and state_size or hidden_size or both must be specified.'
            if state_size and belief_size:
                input_size = state_size + belief_size
            elif state_size:
                input_size = state_size
            else:
                input_size = belief_size
            self.value_model = ValueNetwork(MlpExtractor(
                # We are ignoring the beliefs for now.
                # If this is changed, ValueNetwork class needs to be changed to pass the belief to the MlpExtractor.
                input_size,
                net_arch=[{'vf': [hidden_size for _ in range(value_layers)]}],
                activation_fn=torch.nn.ReLU,
                device=device,
            ), device=device,
                ignore_beliefs=belief_size is None,
                ignore_states=state_size is None)
            self.value_optimizer = torch.optim.Adam(list(self.value_model.parameters()), lr=value_lr)
            self.lr = value_lr
            self.lr_schedule = None
            self.device = device
            self.values_target = None
            self.target_fn = value_target_fn
            self.grad_clip_norm = grad_clip_norm
            self.update_target_every = update_target_every

            self.reset_lr_schedule()
        else:
            self.value_model = None
        self.iteration = 0

    def reset_lr_schedule(self):
        if self.lr_strategy is None:
            return
        self.lr_schedule = self.lr_strategy.reset_schedule(self.value_optimizer)

    @property
    def name(self):
        return self._name

    def values(self, *args):
        if len(args[0].size()) == 3:
            return bottle(self.value_model, args)
        return self.value_model(*args)

    def rewards(self, beliefs, states, actions=None):
        assert len(beliefs.size()) == 3, 'Requires planning horizon dimension.'
        return self._rewards(beliefs, states, actions)

    def _rewards(self, beliefs, states, actions=None):
        return bottle(self.reward_model, (beliefs[1:], states[1:]))

    @property
    def has_scheduler(self):
        return self.lr_strategy is not None

    @property
    def has_value_model(self):
        return self.value_model is not None

    def reset_value_params(self) -> None:
        """
        Reset the weights of the value network (if it exists).
        """
        if self.has_value_model:
            self.reset_lr_schedule()
            reset_model_params(self.value_model)

    def reset_value_target_params(self) -> None:
        """
        Reset the weights of the value *target* network (if it there is a value network).
        """
        if self.has_value_model:
            if self.values_target is None:
                self._update_target_network()
            reset_model_params(self.values_target)

    def train_batch(self, states, embeddings, actions, rewards, dones, gamma, logger):
        if not self.has_value_model:
            return 0.

        value_loss = self._update_value(states, rewards, dones, gamma)

        if self.has_scheduler:
            self.lr_strategy.after_batch(logger)
        return value_loss

    def before_updates(self):
        if self.has_value_model and self.iteration % self.update_target_every == 0:
            self._update_target_network()
        self.iteration += 1

    def after_updates(self, logger):
        if self.has_scheduler:
            self.lr_strategy.after_epoch(logger)

    def _update_value(self, states, rewards, terminals, gamma):
        self.value_model.train()
        with torch.no_grad():
            dones_true = rewards == 0.

            # In the DQN algorithm implementation they do something like this? Mais, pourquoi?
            # next_q_values = torch.cat(self.values_target(next_observations), dim=1)
            # next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
            values_next = bottle(self.values_target,
                                 (states.beliefs[1:].detach(), states.states[1:].detach()))
            target_values = self.target_fn(rewards, dones_true, gamma, values_next)

        # Get current Q-values estimates for each critic network
        # using action from the replay buffer
        current_values = bottle(self.value_model,
                                (states.beliefs[:-1].detach(), states.states[:-1].detach()))

        # Compute critic loss
        value_loss = F.mse_loss(current_values, target_values, reduction='none')

        # Fix terminal states propagating wrong value
        value_loss = self._mask_terminal_states(value_loss, terminals)
        value_loss = value_loss.mean()

        # Optimize the critic
        self.value_optimizer.zero_grad()
        value_loss.backward()
        clip_grad_norm_(self.value_model.parameters(), self.grad_clip_norm, norm_type=2)
        self.value_optimizer.step()
        self.value_model.eval()
        return value_loss.item()

    def _mask_terminal_states(self, value_loss, dones):
        """
        HOTFIX: This is a beautiful example of software engineering. Sometimes you make such a bad mistake that
        adding *more* code instead of improving the original one is the only way forward at the moment...

        So what does this workaround do? We use code that requires a single sequence of states. However, as this code
        actually should work with episodic environments, we actually need two sequences: current_states and next_states.
        Why? Because if we hit a terminal state s_T, this will be in the next_states sequence but not in the current
        one. Similarly, the first state of an episode will never be in the next_states but only in the current_states
        sequence. Now, we cannot simply append or prepend states to one or the other sequence to transform from one into
        the other. If a terminal state is in the middle of the sequence we would also need to merge the start state
        from the middle of the sequence into it. And there's our problem: Now the length of the sequence changes.
        Now, working with tensors usually relies on *matching* shapes -- so we would be running into a lot of trouble.
        In other words, if we wanted to actually use two tensors as in a normal replay buffer setup we need to a lot of
        refactoring. And there's no time for this, right now. So if anyone reads this, now you know why this method
        exists.
        TODO: refactor to fix this (ID1)

        :return:
        """
        terminal_states = torch.cat([torch.zeros((1, *dones.size()[1:]), device=dones.device), dones])[:-1]
        return value_loss[terminal_states == 0]

    def _update_target_network(self):
        if self.values_target is None:
            self.values_target = deepcopy(self.value_model)
        else:
            with torch.no_grad():
                for p in self.values_target.parameters():
                    p.requires_grad = False

                self.values_target.load_state_dict(self.value_model.state_dict())


class ExplorationRewardSpace(RewardSpace):
    def __init__(self,
                 exploration_method: Explorer, reward_model=None,
                 value_lr=None, hidden_size=None, state_size=None, belief_size=None, device=None,
                 value_layers=3, lr_strategy=None, name='',
                 reset_target_every=None, update_target_every=1):
        super().__init__(reward_model,
                         value_lr=value_lr,
                         hidden_size=hidden_size,
                         state_size=state_size,
                         belief_size=belief_size,
                         device=device,
                         value_layers=value_layers,
                         lr_strategy=lr_strategy,
                         name=name,
                         update_target_every=update_target_every)
        self.exploration_method: Explorer = exploration_method
        self.reset_target_every = reset_target_every if reset_target_every > 0 else None

    def _rewards(self, beliefs, states, actions=None):
        assert actions is not None, 'Exploration reward space requires actions to compute reward.'
        return bottle(self.exploration_method.exploration_reward, (beliefs[:-1], states[:-1], actions))
        # if we were to combine the rewards, we would need to do something like this:
        # return self.exploration_method(self.reward_model(...), ...)

    def before_updates(self):
        if self.has_value_model:
            if self.reset_target_every is not None and self.iteration % self.reset_target_every == 0:
                self.reset_value_target_params()
            elif self.iteration % self.update_target_every == 0:
                self._update_target_network()
        # super().before_updates()

    def train_batch(self, states, embeddings, actions, rewards, dones, gamma, logger):
        self.exploration_method.train_batch(rewards, states.beliefs,
                                            states.states,
                                            actions, embeddings)
        return super().train_batch(states, embeddings, actions,
                                   self.rewards(states.beliefs, states.states, actions).unsqueeze(-1), dones, gamma,
                                   logger)


class ValueNetwork(torch.nn.Module):
    def __init__(self, mlp_extractor: MlpExtractor, device,
                 ignore_states=False, ignore_beliefs=True):
        assert not ignore_states or not ignore_beliefs, 'Cannot ignore both states and beliefs.'
        super().__init__()
        self.mlp_extractor = mlp_extractor
        self.lin = torch.nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        self.lin.to(device)
        self.device = device

        self.ignore_states = ignore_states
        self.ignore_beliefs = ignore_beliefs

    def forward(self, beliefs, states):
        # We are ignoring the beliefs for now.
        # If this is changed, the input size of the MlpExtractor needs to be changed too.
        if self.ignore_states:
            features = beliefs
        elif self.ignore_beliefs:
            features = states
        else:
            features = torch.cat([states, beliefs], dim=-1)
        return self.lin(self.mlp_extractor(features)[1])
