Reinforcement Learning in Grid Worlds
========

This repository explores how agents trained with different RL algorithms behave in grid worlds.

## Installation
This project was written with Python 3.8.
All requirements are listed in `requirements.txt`.
It is recommended to set up a virtual Python environment, e.g., with `conda -n mbrl python=3.8`.
Then installing all requirements using:

    pip install -r requirements.txt

## Requirements
All Python dependencies used in this project should be listed in `requirements.txt`.
This list should be a minimal as possible to prevent compatibility and future issues.
If you like to add modules, please make sure you are familiar with [version specifiers](https://www.python.org/dev/peps/pep-0440/#version-specifiers)
so that you can specify the versions required for this project (also to prevent future compatibility issues).

Non-Python dependencies should be described in `readme.md`.

## Running
The experiments can be run via the CLI.
[Hydra](https://hydra.cc/) is used to manage configuration.
The default parameters can be found in `experiments/base.yaml`.
To use other parameters they can be passed as an argument, e.g.:

    python main.py mpc=value

This runs the PlaNet algorithm with a value network.
Individual parameters can be set as described in the [Hydra docs](https://hydra.cc/docs/advanced/override_grammar/basic):

    python main.py mpc=value mpc.planning_horizon=6

Check the `experiments` directory for all parameters.
Read the following subsections for advanced configuration.

**Note:**
Hydra also [changes the working directory](https://hydra.cc/docs/tutorials/basic/running_your_app/working_directory/) to
`outputs/<date>/<execution time>_<job name>` so all saved outputs can be found here.

### Presets
There are also presets for experiments/algorithms that require many parameters to be changed.
These can be found in the top-level folder of the `experiments` directory and loaded with `--config-name`, for example:

    python --config-name dreamer

### Sweeping Parameters
Hydra also allows to start multiple experiments by sweeping parameters.
This is called [multi-run](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run).
We use the [JobLib Launcher](https://hydra.cc/docs/plugins/joblib_launcher) so that all experiments are run in *parallel*.
Which might be too much on your memory.
In this case comment out following line in `experiments/base.yaml`:

      - override hydra/launcher: joblib

To use multi-run, simply prepend set the flag `--multirun` (or just `-m`) to the arguments in the command line and 
specify the parameters you want to sweep over using a comma separated list.
Usually, this will be the seed, for example:

    python main.py --multirun environment.seed=1,2,3 mpc=value

### Naming Experiments
Finally, you might want to give your experiment a name, so that you can find it more easily than by date and time.
The name will also occur in the output directory.
The default job name is *main*.
You can change it by specifying the `hydra.job.name` parameter, e.g.:

    python main.py hydra.job.name=PlaNet

## Experimenting

### Applying exploration
Use the parameter `+exploration=state` to use exploration reward based on state prediction uncertainty.

### Loading a trained model
Trained models can be loaded by setting the `load` parameter.
Since this parameter is not part of the default hydra config, it needs to be added, e.g.

    python main.py +load=/path/to/model.zip

If the load path is empty, i.e., `+load=`, the path can be specified interactively in the CLI.

### Rendering evaluation episodes
Another non-default parameter is the `render` option.
By adding the argument `+evaluation.render=1` when running the program, the final evaluation episodes will be rendered.
Note, that this requires a display.

### Visualizing the replay buffer
To visualize the replay buffer the `visualize.py` script can be used.
Basically, it requires the environment from which the data is generated, the respective seed (if applicable),
and the path to the stored replay buffer:

    python visualize.py --env MiniGrid-FourRooms-v0 --env_seed 123 --path /path/to/replay_buffer

Since the replay buffer may include many episodes, the `--range` option can be set to only visualize the only render the
given range.
The `--interactive` flag is useful, when running the commands multiple times with varying options.
It will ask for the missing parameters.
For example, if the range is not set, the program will ask for the range in the CLI.

All parameters and their descriptions can be shown with the `--help` flag.

## Tests
**Note:** The current version of this repository contains **no** automated tests :(

The code is tested using the [`pytest` framework](https://docs.pytest.org/en/6.2.x/).
All tests can be found in the `tests` module.
To run the tests execute

    python -m pytest

which should automatically discover all tests.

Before committing to the `master` branch all tests must be passing (green).

## Licensing
This repository strongly builds on the code of [dreamer-pytorch](https://github.com/zhaoyi11/dreamer-pytorch) published under the MIT license.
Some code is also taken from [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) published under the MIT license.

We tried to comment all those parts of the code that have been taken from other sources.
Due to refactorings it could be that the original code was split up into further methods, classes, or documents which may
not contain the comment with the direct reference.

We tried to ensure that all 3rd party code that was used it published under the MIT license.
However, we found it impossible to track back the sources entirely.
Thus, we cannot ensure that this project fulfills the MIT license requirements.
Nevertheless, we license all our *original* code, i.e., such code we wrote ourselves and published it in this project for the first time, under the MIT license.

## References
This project implements the PlaNet [1] algorithm, as well as parts of the Dreamer [2] and Plan2Explore [3] algorithms:

1. Hafner, D., Lillicrap, T.P., Fischer, I.S., Villegas, R., Ha, D.R., Lee, H., & Davidson, J. (2019). Learning Latent Dynamics for Planning from Pixels. ArXiv, abs/1811.04551.
2. Hafner, D., Lillicrap, T.P., Ba, J., & Norouzi, M. (2020). Dream to Control: Learning Behaviors by Latent Imagination. ArXiv, abs/1912.01603.
3. Sekar, R., Rybkin, O., Daniilidis, K., Abbeel, P., Hafner, D., & Pathak, D. (2020). Planning to Explore via Self-Supervised World Models. ArXiv, abs/2005.05960.
