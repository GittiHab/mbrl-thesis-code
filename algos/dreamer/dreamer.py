from algos.PlaNet.planet import PlaNet
from algos.dreamer.policies import DreamerPolicy


class Dreamer(PlaNet):
    def __init__(self, *args, **kwargs):
        assert 'policy' in kwargs and isinstance(kwargs['policy'], DreamerPolicy) or \
               isinstance(args[0], DreamerPolicy), 'Policy needs to be instance of DreamerPolicy.'
        super().__init__(*args, **kwargs)

    def _setup_optimizers(self):
        self.add_value_optimizer(self.policy)

    def train(self, gradient_steps: int, batch_size: int) -> None:
        return super().train(gradient_steps, batch_size)