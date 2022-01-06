def figure_to_PIL(fig):
    # From https://stackoverflow.com/a/61754995
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    from PIL import Image
    buffer = io.BytesIO()
    fig.savefig(buffer)
    buffer.seek(0)
    image = Image.open(buffer)
    return image


def get_git_revision_hash():
    # Source: https://stackoverflow.com/a/21901260
    import subprocess
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()


def get_git_branch():
    # Source: https://stackoverflow.com/a/21901260
    # and https://stackoverflow.com/questions/6245570/how-to-get-the-current-branch-name-in-git
    import subprocess
    return subprocess.check_output(['git', 'branch', '--show-current']).decode('ascii').strip()


### START: Following code (get_parameters and FreezeParameters) is taken from
### dreamer-pytorch by juliusfrost published under the MIT license.
### https://github.com/juliusfrost/dreamer-pytorch/blob/47bd509ab5cffa95ec613fd788d7ae1fe664ecd5/dreamer/utils/module.py
### Code has been modified.

def get_parameters(modules):
    """
    Given a list of torch modules, returns a list of their parameters.
    :param modules: iterable of modules
    :returns: a list of parameters
    """
    model_parameters = []
    for module in modules:
        model_parameters += list(module.parameters())
    return model_parameters


class FreezeParameters:
    def __init__(self, modules, parameters):
        """
        Context manager to freeze parameters of given modules.
        :param modules: iterable of modules.
        """
        self.modules = modules
        self.parameters = parameters
        self.param_states = [p.requires_grad for p in get_parameters(self.modules)]
        self.param_states.extend([p.requires_grad for p in self.parameters])

    def __enter__(self):
        for param in get_parameters(self.modules):
            param.requires_grad = False
        for param in self.parameters:
            param.requires_grad = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        i = 0
        for i, param in enumerate(get_parameters(self.modules)):
            param.requires_grad = self.param_states[i]
        for j, param in enumerate(self.parameters):
            param.requires_grad = self.param_states[i + j]


### END taken from dreamer-pytorch


def total_timesteps(cfg, warmup=True, exploration=True):
    total = cfg.environment.steps
    if exploration and 'exploration' in cfg and 'steps' in cfg.exploration:
        total += cfg.exploration.steps
    if warmup:
        total += cfg.training.warmup_steps
    return total


def reset_model_params(model):
    # https://discuss.pytorch.org/t/reset-model-weights/19180/6
    layers = list(model.children())
    while layers:
        layer = layers.pop()
        layers.extend(layer.children())
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
