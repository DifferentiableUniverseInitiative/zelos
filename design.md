# Design document

## Goal

Our goal is to define a standardized interface and set of tools for training,
sharing, and using emulators of the matter power spectrum.

The project would be deemed successful when practically nobody ever needs to
actually run a Boltzmann code anymore.

## Related work

- [PICO](https://arxiv.org/pdf/0712.0194.pdf)
- [pyPICO](https://github.com/marius311/pypico)

...

## Post-mortem of previous emulators

In this section we try to identify why previous general purpose emulators
may not have become more widespread.

### PICO (comments from Marius)

PICO does a global polynomial interpolation in ~10 dimensional parameter space (6 LCDM + ~4 extensions), accurate within a region which roughly covers the _union_ of WMAP and SPT-TT 5σ contours. It is differentiable (although this feature was undocumented).

PICO could be regarded as one of the more succesful emulators as it's the only one which got some built-in support to CosmoMC. To my knowledge, it was very useful to a few people within in Planck during the analysis, has been used in some classes, and gotten the occasional use here and there. It was not used in the final "grid" of Planck chains, which were run with CAMB. It did not otherwise achieve "widespread" use. Some reasons which may have caused this relevant for this project are:

1. Although its accuracy is very good (good enough for Planck), this was not very well documented as I never got around to writing another paper specifically for the most recent trainings I did. I got the sense that people felt compelled to verify the accuracy themselves. 

2. While the training region was big (5σ _WMAP/SPT_ is huge by today's standards), it was not big enough that for some crazy dataset you were testing, or for a normal one but just a few samples in your chain, you didn't leave the training region. 

3. Chains are often far less useful unless they contain "derived" parameters, which PICO does not calculate. 

4. Previous three points combine to say that your analysis code _needed_ a CAMB fallback anyway. Given this, I think the pain of setting up PICO and the logic of switching was often not worth the pain for people rather than just launching CAMB chains and just waiting longer. 

5. Thanks to curse of dimensionality, you probably can't ever train all the extensions people want simultaenously (probably ~20 "extension" parameters in the Planck grid). So you pick subsets and have "datafiles" which people need to remember to switch between for given runs. 


Mistake (1) just needs to not be repeated. (5) can probably be alleviated with a better code interface that switches for you. (2) can perhaps be alleviated with smarter emulation and or if your target isn't the CMB, or isn't as big as WMAP/SPT. (3) is fundamental and problematic.


Finally, I will mention that training is highly non-trivial in that high dimensions and if your goal is to be accurate enough for Planck. This is of course specific to our polynomial interpolation, but required lots of careful handtuning of the training region (which required running CAMB chains on the targed data in the first place) and picking a physically-motivated transformation of the input and output parameters (e.g. training in θs the sound horizon angular scale as opposed to H₀). Providing a system that makes it trivial for users to compute their own trainings would be effectively impossible in the PICO setup. 



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
