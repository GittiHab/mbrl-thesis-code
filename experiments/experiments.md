Experiments Planned
=======

This is an overview of the planned experiments.

All experiments with exploration phases should store intermediate models:
`+checkpointing=freq checkpointing.every=500`

## Four Rooms, Seed 10 and 18
Base parameters: `environment=fourrooms environment.env_seed=10 mpc.planning_horizon=7 mpc.candidates=500 mpc.top_candidates=50 training.warmup_steps=2000 mpc=value`

### Update after 1 episode
| Name              | Parameters                              | Notes | Results | Result path (if executed) |
|-------------------|-----------------------------------------|-------|---------|---------------------------|
| A2C Baseline      | `algorithm=a2c environment.steps=30000` |       |         |                           |
| PlaNet Baseline   | `environment.steps=28000`     |       |         |                           |
| Explore State     | `exploration=state exploration.steps=8000 environment.steps=20000` |       |         |                           |
| Explore Embedding | `exploration=observation exploration.steps=8000 environment.steps=20000` |       |         |                           |
### Update after 1 time step
| Name              | Parameters                              | Notes | Results | Result path (if executed) |
|-------------------|-----------------------------------------|-------|---------|---------------------------|
| Baseline 1-step | `environment.steps=28000 training=step mpc.update_target_every=100` |       |         |                           |
| Explore State 1-step | `exploration=observation exploration.steps=8000 environment.steps=20000 training=step mpc.update_target_every=100 exploration.reset_target_every=300` |       |         |                           |

## Exploration vs. Unguided
Base parameters: *same as above*

The experiment investigates how an agent with curiosity reward behaves compared to an agent *without any* reward signal.
Ideally, the one with the exploration rewards visits more unique states than the other one,
who is expected to mainly get stuck.

| Name              | Parameters                              | Notes | Results | Result path (if executed) |
|-------------------|-----------------------------------------|-------|---------|---------------------------|
| A2C Baseline      | `algorithm=a2c environment.steps=10000 environment.muted=true` |       |         |                           |
| PlaNet no signal   | `mpc=value environment.steps=10000 environment.muted=True`     |       |         |                           |
| Explore State     | `mpc=value exploration=state exploration.steps=10000 environment.steps=0` |       |         |                           |
| Explore Embedding | `mpc=value exploration=observation exploration.steps=10000 environment.steps=0` |       |         |                           |