ETH Machine Learning Projects
=============================

This repository contains the framework for the practical projects offered
during the *Machine Learning* course at ETH Zurich.

Getting Started
---------------

First you need to install miniconda_ on your system.

.. _miniconda: https://conda.io/docs/install/quick.html#linux-miniconda-install

Having installed miniconda, clone the repository and run the setup script:

.. code-block:: shell

    git clone https://gitlab.vis.ethz.ch/vwegmayr/ml-project.git
    cd ml-project
    python setup.py

Running an example experiment
-----------------------------

Make sure the environment is activated:

.. code-block:: shell

    source activate ml_project

To run an example experiment, simply type

.. code-block:: shell

    $ smt run config .example_config.yaml -X data/X_train.npy 

    >>>=========== Config ===========
    >>>{'action': 'fit_transform',
    >>>'class': <class 'ml_project.models.transformers.RandomSelection'>,
    >>>'params': {'n_components': 1000, 'random_state': 37}}
    >>>==============================

    >>>Record label for this run: '20170810-131658'
    >>>Data keys are [20170810-131658/RandomSelection.pkl(9b028327c83a153c0824ca8701f3b78a5106071c [2017-08-10 13:17:04]), 20170810-131658/X_new.npy(b8c093d7c8e13399b6fe4145f14b4dbc0f241503 [2017-08-10 13:17:04])]

The default experiment will reduce the dimensionality of the training data by selecting 1000
dimensions at random. For the experiment shown above, you would find the results
in `data/20170810-131658`. It produced two outputs, first the fitted model
*RandomSelection.pkl* and second the transformed training data *X_new.npy*.

You can choose from different examples in the `.example_config` file.


