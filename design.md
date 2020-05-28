# Design document

## Goal

Our goal is to define a standardized interface and set of tools for training,
sharing, and using emulators of the matter power spectrum.

The project would be deemed successful when practically nobody ever needs to
actually run a Boltzmann code anymore.

## Related work

- [Pico](https://arxiv.org/pdf/0712.0194.pdf)
- [pyPico](https://github.com/marius311/pypico)

...

## Post-mortem of previous emulators

In this section we try to identify why previous general purpose emulators
may not have become more widespread.


## Guiding principles

- The emulator code should be trivially installable, no complex dependencies, no
compilation

- Emulators should be accessible through various APIs: Python, Julia, JAX, TensorFlow

- As much as possible, emulators should be differentiable

- Emulators should be easily re-trainable by the user with tweaked settings (even if potentially costly)


## Proposed design

### A simple user-facing interface

The installation process shouldn't  be more complicated than
```bash
$ pip install zelos
```

For most users, who just want to grab a power spectrum, all they should have to
do is:
```python
import zelos
# This automatically downloads the emulator from the cloud
emulator = zelos.get_emulator('hub:CAMB_linear_matter_power')
pk = emulator(params, k, z) # Evaluates the emulator for a set of params at desired k and z
```

If accessed from a framework that support  autodiff, like JAX or Julia/Zygote, we
should be able to do the following:
```python
def pk(params):
  return emulator(params, k, z)

jax.grad(pk)(params)
```

### A standard to describe an emulator

We want to  be able to automatically train/retrain emulators, and make it easy
for  people to tweak and retrain any emulators. Also very important that their
should never be any ambiguity on what exact code version and parameter settings
an emulator is emulating.

So we need some sort of abstraction of an actual emulator that contains all the
information to recreate it. For instance we could use some sort of YAML file:
```YAML
name: CAMB_linear_matter_power
author: EiffL
# All emulators should be defined from a container
container: eiffl/CAMB:v0.1
# configuration settings for the code
config: config.ini

# Some specification of the emulator, and all params of that emulator
emulator_fn:
  - type: zelos.emulators.neural_network
  - params:
    - n_layers: 5
    - layer_size: 128

# Some parameters that define the training
training:
  - type: RMSE_minimization
  - params:
    - optimizer: ADAM
...


# These will be the inputs of the emulator
parameters:
  - Omega_b: [0.01, 0.05]
  - w0: [-3, -0.3]
...

# Defines which quantities will be emulated
outputs:
  - linear_matter_power:
    - k: [1e-4, 1e2]
    - z: [10., 0.]
```

This may require to have a small driver that knows how to run CAMB for instance
from the settings provided in this file and given the Docker container.


### Automatically building emulators

Given the YAML description of the emulator, the following should work:
```bash
$ zelos_build CAMB_linear_matter_power.yaml
>>> Retrieving container eiffl/CAMB:v0.1 ...... [done]
>>> Building training set ......................[done]
>>> Training model .............................[done]
>>> Evaluating accuracy ........................[done]

Report:
-------
Trained emulator for CAMB_linear_matter_power.yaml in 3000s
Maximum relative error: 0.5e-3
Exporting emulator to CAMB_linear_matter_power.tar.gz
$ ls
CAMB_linear_matter_power.tar.gz
```
This command line tool should  be able  to generate an archive of the emulator
that only contains the "weights" of the model, the YAML file used for training
and a report file that contains stats on accuracy of the model, when and where
it was trained, etc.


Then, if satisfied with the result, the YAML and/or the archive can be pushed
to the online repo.
