import torch
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
import gym
from wrappers import SymbolicWrapper, SimpleActionsWrapper

__all__ = ['TrueWorldModel', 'TrueModel', 'TrueTransitionModel', 'TrueTerminalModel', 'TrueRewardModel']


class TrueWorldModel:
    def save_state(self):
        pass

    def load_state(self, state):
        pass

    def reset(self):
        pass

    def __init__(self, env):
        self.transition_model = TrueTransitionModel(env)
        self.reward_model = TrueRewardModel(env)
        self.pcont_model = TrueTerminalModel(env)


class TrueModel:
    def __init__(self, env_id):
        '''
        :param env_id: Requires gym_minigrid environments
        '''
        self._env_id = env_id

        env = gym.make(env_id)
        env = SymbolicWrapper(SimpleActionsWrapper(env))
        env.reset()
        self._env = env

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def eval(self):
        pass

    def train(self):
        pass

    def state_dict(self):
        return {}

    def parameters(self):
        return []


class TrueTransitionModel(TrueModel):
    def forward(self, states, actions, belief=None):
        '''

        @param states:
        @param actions:
        @param belief:
        @return: The next states when taking the given actions in the given states.
        '''
        squeeze = False
        if len(states.size()) == 1:
            squeeze = True
            states = states.unsqueeze(0)
            actions = actions.unsqueeze(0)
            actions = actions.argmax(-1)

        def create_env_factory(state, env_id):
            def env_factory():
                env = gym.make(env_id)
                env.reset()
                env.unwrapped.agent_dir = state[-1].cpu().item()
                env.unwrapped.agent_pos = state[:-1].cpu().numpy().tolist()
                env = SymbolicWrapper(SimpleActionsWrapper(env))
                return env

            return env_factory

        env = DummyVecEnv([create_env_factory(s, self._env_id) for s in states])
        states = env.step(actions.cpu().numpy())[0]
        if squeeze:
            states = states.squeeze(0)
        return None, states


class TrueRewardModel(TrueModel):
    def forward(self, state, belief=None):
        from gym_minigrid.minigrid import Goal

        squeeze = False
        if len(state.size()) == 1:
            squeeze = True
            state = state.unsqueeze(0)

        rewards = []
        state_np = state.cpu().numpy().astype(int)
        for i in range(state.size(0)):
            rewards.append(int(isinstance(self._env.unwrapped.grid.get(*state_np[i][:2]), Goal)))
        rewards = torch.tensor(rewards, dtype=state.dtype, device=state.device)
        if squeeze:
            rewards = rewards.squeeze(0)
        return rewards


class TrueTerminalModel(TrueModel):
    def forward(self, state, belief=None):
        from gym_minigrid.minigrid import Goal

        squeeze = False
        if len(state.size()) == 1:
            squeeze = True
            state = state.unsqueeze(0)

        non_terminals = []
        state_np = state.cpu().numpy().astype(int)
        for i in range(state.size(0)):
            non_terminals.append(not isinstance(self._env.unwrapped.grid.get(*state_np[i][:2]), Goal))
        non_terminals = torch.tensor(non_terminals, dtype=state.dtype, device=state.device)

        if squeeze:
            non_terminals = non_terminals.squeeze(0)
        return non_terminals
