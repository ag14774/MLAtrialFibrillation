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

    >> =========== Config ===========
    >> {'action': 'fit_transform',
    >> 'class': <class 'ml_project.models.transformers.RandomSelection'>,
    >> 'params': {'n_components': 1000, 'random_state': 37}}
    >> ==============================

    >> Record label for this run: '20170810-131658'
    >> Data keys are [20170810-131658/RandomSelection.pkl(9b028327c83a153c0824ca8701f3b78a5106071c [2017-08-10 13:17:04]),
    >> 20170810-131658/X_new.npy(b8c093d7c8e13399b6fe4145f14b4dbc0f241503 [2017-08-10 13:17:04])]

The default experiment will reduce the dimensionality of the training data by
selecting 1000 dimensions at random.

For the experiment shown above, you would find the results in
:code:`data/20170810-131658`.

It produced two outputs, first the fitted model *RandomSelection.pkl* and second
the transformed training data *X_new.npy*.

To view the experiment record, type :code:`smtweb`.

This command will open a new window in your webbrowser, where you can explore
the information stored about the example experiment.

You can choose from different examples in the `.example_config` file.

More details on experiments
---------------------------

.. _Sumatra: https://pythonhosted.org/Sumatra/

Let us consider the command from before in more detail:

.. code-block:: shell

    $ smt run config .example_config.yaml -X data/X_train.npy 

The first part :code:`smt` invokes Sumatra_, which is an experiment tracking tool.

The second part :code:`run` tells Sumatra to execute the experiment runner.

The argument :code:`config` informs the runner about the experiment type, more about this later.

Now there are only input arguments left such as config file and data.

In addition to :code:`config` experiments, you can run :code:`model` experiments.

These two cover fit/fit_transform and transform/predict, respectively.

The reason is that for fit/fit_tranform you typically require parameters, whereas
for transform/predict you start from a fitted model.

Continuing the example, we can transform the test data, using
the fitted model from before:

.. code-block:: shell

    $ smt run model data/20170810-131658/RandomSelection.pkl -X data/X_test.npy -a transform
    >> Record label for this run: '20170810-134027'
    >> Data keys are [20170810-134027/X_new.npy(b33b0e0b794b64e5d284a602f5440620a21cac1c [2017-08-10 13:40:32])]

The experiment type is :code:`model` now. The input arguments are the model to
use (the one we created earlier), the input data,

and finally the action that the model should perform on the input
(:code:`transform` in this case).

Again, Sumatra created an experiment record, which you can use to track input/output paths.


More tools
----------

.. _kaggle-cli: https://github.com/floydwch/kaggle-cli

A very convenient tool included in the ml-project framework is kaggle-cli_.
It can be used to submit predictions to kaggle and to view previous submissions.