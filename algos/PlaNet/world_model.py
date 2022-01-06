from models.models import Encoder, ObservationModel, RewardModel, TransitionModel, PCONTModel
from torch import optim
import torch
from gym import Env

__all__ = ['DreamerModel']

from type_aliases import StatePredictions


class DreamerModel:
    # This class is based on code from dreamer-pytorch by zhaoyi11
    # https://github.com/zhaoyi11/dreamer-pytorch/blob/500ae4fcd7143342b1eb77835e39c5d1d4986170/agent.py#L48

    @staticmethod
    def from_args(args, device, env: Env = None):
        action_size = env.action_space.n if env else args.action_size
        observation_size = env.observation_space.shape[0] if env else args.observation_size
        return DreamerModel(observation_size,
                            args.belief_size,
                            args.state_size,
                            action_size,
                            args.hidden_size,
                            args.embedding_size,
                            args.dense_act,
                            args.cnn_act,
                            args.pcont,
                            args.world_lr,
                            device,
                            args.symbolic)

    def __init__(self,
                 observation_size,
                 belief_size,
                 state_size,
                 action_size,
                 hidden_size,
                 embedding_size,
                 dense_act,
                 cnn_act,
                 pcont,
                 world_lr,
                 device,
                 symbolic):
        self.observation_size = observation_size
        self.belief_size = belief_size
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.dense_act = dense_act
        self.cnn_act = cnn_act
        self.pcont = pcont
        self.world_lr = world_lr

        self.symbolic = symbolic
        self.device = device

        self.reset()

    def eval(self):
        self.transition_model.eval()
        self.observation_model.eval()
        self.reward_model.eval()
        self.encoder.eval()

    def train(self):
        self.transition_model.train()
        self.observation_model.train()
        self.reward_model.train()
        self.encoder.train()

    def save_state(self):
        return {'transition_model': self.transition_model.state_dict(),
                'observation_model': self.observation_model.state_dict(),
                'reward_model1': self.reward_model.state_dict(),
                'encoder': self.encoder.state_dict(),
                'world_optimizer': self.world_optimizer.state_dict()
                }

    def load_state(self, model_data):
        self.transition_model.load_state_dict(model_data['transition_model'])
        self.observation_model.load_state_dict(model_data['observation_model'])
        self.reward_model.load_state_dict(model_data['reward_model1'])
        self.encoder.load_state_dict(model_data['encoder'])
        self.world_optimizer.load_state_dict(model_data['world_optimizer'])

    def reset(self):
        self.transition_model = TransitionModel(
            self.belief_size,
            self.state_size,
            self.action_size,
            self.hidden_size,
            self.embedding_size,
            self.dense_act).to(device=self.device)

        self.observation_model = ObservationModel(
            self.symbolic,
            self.observation_size,
            self.belief_size,
            self.state_size,
            self.embedding_size,
            activation_function=(self.dense_act if self.symbolic else self.cnn_act)).to(
            device=self.device)

        self.reward_model = RewardModel(self.belief_size,
                                        self.state_size,
                                        self.hidden_size,
                                        self.dense_act).to(device=self.device)

        self.encoder = Encoder(
            self.symbolic,
            self.observation_size,
            self.embedding_size,
            self.cnn_act).to(device=self.device)

        self.pcont_model = PCONTModel(
            self.belief_size,
            self.state_size,
            self.hidden_size,
            self.dense_act).to(device=self.device)

        # setup the paras to update
        self.world_param = list(self.transition_model.parameters()) \
                           + list(self.observation_model.parameters()) \
                           + list(self.reward_model.parameters()) \
                           + list(self.encoder.parameters())
        if self.pcont:
            self.world_param += list(self.pcont_model.parameters())

        # setup optimizer
        self.world_optimizer = optim.Adam(self.world_param, lr=self.world_lr)

    def infer_state(self, observation, action, belief, state_prev, return_all=False):
        # Action and observation need extra time dimension
        state_prediction: StatePredictions = self.transition_model(
            state_prev,
            action.unsqueeze(0),
            belief,
            self.encoder(observation.type(torch.float)).unsqueeze(0))

        if return_all:
            return state_prediction
        # Remove time dimension from belief/state
        belief, posterior_state = state_prediction.beliefs.squeeze(0), state_prediction.posterior_states.squeeze(0)
        return belief, posterior_state
