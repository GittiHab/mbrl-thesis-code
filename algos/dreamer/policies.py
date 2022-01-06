import torch.nn as nn
from gym import Env
from torch.optim import Adam
import torch
from algos.PlaNet.optimizer import Optimizer
from algos.PlaNet.reward_space import RewardSpace, setup_reward_space
from algos.PlaNet.world_model import DreamerModel
from algos.commons.policies import ModelBasedPolicy
from models.model_params import ModelParams
from models.models import bottle, ActorModel
from type_aliases import StatePredictions, ImaginedTrajectories
from utils import FreezeParameters


class DreamerPolicy(ModelBasedPolicy, Optimizer):

    @staticmethod
    def from_config(config, model_params, env: Env, world_model: DreamerModel, lr_step_size=100):
        value_space = setup_reward_space(config, world_model, lr_step_size, 'value_real')  # TODO
        return DreamerPolicy(world_model, value_space, config.planning_horizon, model_params,
                             hasattr(env.action_space, 'n'), env.observation_space, env.action_space, config.actor_lr)

    def __init__(self, world_model: DreamerModel, value_space: RewardSpace, planning_horizon,
                 model_params: ModelParams, discrete: bool, observation_space, action_space,
                 learning_rate: int = 8e-5, grad_clip_norm=100.0, **kwargs):
        super().__init__(observation_space, action_space, **kwargs)

        self.planning_horizon = planning_horizon
        self.grad_clip_norm = grad_clip_norm
        self.discrete = discrete

        self.value_space: RewardSpace = value_space
        self.reward_space: RewardSpace = RewardSpace(world_model.reward_model)
        self.world_model = world_model

        self.actor_model = self._create_actor_network(model_params.action_size, model_params.belief_size,
                                                      model_params.state_size, model_params.hidden_size,
                                                      model_params.dense_act, discrete, model_params.device)
        self.actor_optimizer = Adam(self.actor_model.parameters(), lr=learning_rate)
        self.reset_state()

    @staticmethod
    def _create_actor_network(action_size, belief_size, state_size, hidden_size, dense_act, discrete, device):
        return ActorModel(
            action_size,
            belief_size,
            state_size,
            hidden_size,
            activation_function=dense_act,
            discrete=discrete).to(device=device)

    def set_reward_space(self, reward_space: RewardSpace):
        self.reward_space = reward_space

    @property
    def name(self):
        return 'DreamerPolicy'

    @property
    def has_scheduler(self):
        # TODO: maybe add one...
        return False

    def train_batch(self, states: StatePredictions, embeddings, actions, rewards, dones, gamma, logger):
        return self.update_actor(states.beliefs, states.posterior_states, gamma)

    def before_updates(self):
        return self.value_space.before_updates()

    def _action_for(self, belief, state, deterministic):
        action, _ = self.actor_model(belief, state, deterministic=deterministic, with_logprob=False)
        return action

    def output_action(self, action):
        if self.discrete:
            return action.argmax(-1)
        return action

    def update_actor(self, beliefs, posterior_states, gamma):
        # This method is strongly based on code from dreamer-pytorch by zhaoyi11 published under the MIT license.
        # https://github.com/zhaoyi11/dreamer-pytorch/blob/500ae4fcd7143342b1eb77835e39c5d1d4986170/agent.py#L249

        # freeze params to save memory
        with FreezeParameters([self.value_space.value_model], self.world_model.world_param):
            # latent imagination
            imagined = self._latent_imagination(beliefs, posterior_states)  # TODO: with logprob?

            # update actor
            if imagined.states.grad_fn is not None or imagined.logprobs is not None:
                # if our actor has no gradients, don't waste our time on this
                actor_loss = self._compute_loss_actor(
                    imagined.beliefs, imagined.states, imagined.rewards, imagined.logprobs, gamma)

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                # from utils import plot_grad_flow
                # plot_grad_flow(self.actor_model.named_parameters())
                nn.utils.clip_grad_norm_(self.actor_model.parameters(), self.grad_clip_norm, norm_type=2)
                self.actor_optimizer.step()
                actor_loss = actor_loss.item()
            else:
                actor_loss = 0.

        # update critic
        # TODO: Add a p_cont network to predict the probability that the episode continues after this state:
        p_cont = torch.ones_like(imagined.rewards)
        critic_loss = self.value_space.train_batch(imagined.detach(), None, imagined.actions[1:], imagined.rewards,
                                                   1 - p_cont, gamma)

        return actor_loss + critic_loss

    def _latent_imagination(self, beliefs, posterior_states, with_logprob=False):
        # This method is taken from dreamer-pytorch by zhaoyi11 published under the MIT license.
        # https://github.com/zhaoyi11/dreamer-pytorch/blob/500ae4fcd7143342b1eb77835e39c5d1d4986170/agent.py#L214
        # Code has been modified.

        # Rollout to generate imagined trajectories
        device = posterior_states.device
        dimensions = posterior_states.size()
        if len(dimensions) == 3:
            chunk_size, batch_size, _ = list(dimensions)  # flatten the tensor
            flatten_size = chunk_size * batch_size

            posterior_states = posterior_states.detach().reshape(flatten_size, -1)
            beliefs = beliefs.detach().reshape(flatten_size, -1)

        imag_beliefs, imag_states, imag_actions, imag_ac_logps = [beliefs], [posterior_states], [], []

        for i in range(self.planning_horizon + 1):
            imag_action, imag_ac_logp = self.actor_model(
                imag_beliefs[-1].detach(),
                imag_states[-1].detach(),
                deterministic=False,
                with_logprob=with_logprob or imag_states[-1].grad_fn is None,
            )
            imag_actions.append(imag_action)
            imag_action = imag_action.unsqueeze(dim=0)

            imag_belief, imag_state = self.world_model.transition_model(imag_states[-1], imag_action, imag_beliefs[-1])
            imag_beliefs.append(imag_belief.squeeze(dim=0))
            imag_states.append(imag_state.squeeze(dim=0))

            if i < self.planning_horizon:
                if with_logprob or imag_states[-1].grad_fn is None and imag_ac_logp is not None:
                    imag_ac_logps.append(imag_ac_logp.squeeze(dim=0))

        imag_beliefs = torch.stack(imag_beliefs, dim=0).to(
            device)  # shape [horizon+1, (chuck-1)*batch, belief_size]
        imag_states = torch.stack(imag_states, dim=0).to(device)
        imag_actions = torch.stack(imag_actions, dim=0).to(device)

        if with_logprob or imag_states.grad_fn is None and len(imag_ac_logps) > 0:
            imag_ac_logps = torch.stack(imag_ac_logps, dim=0).to(device)
        else:
            imag_ac_logps = None

        with torch.no_grad():
            imag_rewards = self.reward_space.rewards(imag_beliefs[1:], imag_states[1:], imag_actions)  # TODO: actions not aligned correctly

        return ImaginedTrajectories(beliefs=imag_beliefs[:-1],
                                    states=imag_states[:-1],
                                    actions=imag_actions,
                                    rewards=imag_rewards.unsqueeze(-1),
                                    logprobs=imag_ac_logps)

    def _compute_loss_actor(self, imag_beliefs, imag_states, imag_rewards, imag_ac_logps, gamma):
        # This method is taken from dreamer-pytorch by zhaoyi11 published under the MIT license.
        # https://github.com/zhaoyi11/dreamer-pytorch/blob/500ae4fcd7143342b1eb77835e39c5d1d4986170/agent.py#L168
        # Code has been modified.

        # reward and value prediction of imagined trajectories
        imag_values = self.value_space.values(imag_beliefs[1:], imag_states[1:])

        with torch.no_grad():
            # if self.args.pcont:
            #     pcont = bottle(self.world_model.pcont_model, (imag_beliefs, imag_states))
            # else:
            pcont = gamma * torch.ones_like(imag_rewards)
        pcont = pcont.detach()

        returns = self.value_space.target_fn(imag_rewards, torch.zeros_like(imag_rewards), gamma, imag_values)

        discount = torch.cumprod(torch.cat([torch.ones_like(pcont[:1]), pcont[:-1]], 0), 0).detach()

        if imag_states.grad_fn is None:
            returns *= imag_ac_logps

        actor_loss = -torch.mean(discount[:-1] * returns)
        return actor_loss
