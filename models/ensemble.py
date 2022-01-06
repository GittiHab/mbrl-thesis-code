import torch
from torch import nn


class Ensemble(nn.Module):

    def __init__(self, model_class, params, instances):
        super().__init__()
        self._models = nn.ModuleList([model_class(*params) for _ in range(instances)])

    def save(self, filepath=None):
        data = [model.state_dict() for model in self._models]
        if filepath is not None:
            torch.save(data, filepath)
        return data

    def load(self, filepath=None, data=None):
        if filepath is not None:
            data = torch.load(filepath)
        elif data is None:
            raise Exception('Either filepath or data must be given.')
        [model.load_state_dict(data[i]) for i, model in enumerate(self._models)]


class STDEnsemble(Ensemble):

    def forward(self, *x):
        xs = torch.stack(self.forward_single(*x), dim=0)
        variance = torch.var(xs, dim=0)
        if len(variance.size()) > 1:
            return variance.mean(dim=-1)
        return variance

    def forward_single(self, *x):
        return [m(*x) for m in self._models]
