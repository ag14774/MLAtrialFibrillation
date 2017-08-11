ETH Machine Learning Projects
=============================

.. _scikit-learn: http://scikit-learn.org/stable/
.. _sumatra: https://pythonhosted.org/Sumatra/
.. _miniconda: https://conda.io/docs/install/quick.html

This repository contains the framework for the practical projects offered
during the *Machine Learning* course at ETH Zurich. It serves two main purposes:

* Convenient execution of machine learning models conforming to the scikit-learn_ pattern.
* Structured & reproducible experiments by integration of sumatra_ and miniconda_.

Getting Started
---------------

First you need to install miniconda_ on your system.

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

    $ smt run --config .example_config.yaml -X data/X_train.npy -a fit_transform

    >> =========== Config ===========
    >> {'class': <class 'ml_project.models.transformers.RandomSelection'>,
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

Let us consider the command from before in more detail:

.. code-block:: shell

    $ smt run --config .example_config.yaml -X data/X_train.npy -a fit_transform

* :code:`smt` invokes sumatra_, which is an experiment tracking tool.

* :code:`run` tells sumatra to execute the experiment runner.

* :code:`--config` points to the paramter file for this experiment.

* :code:`-X` points to the input data

* :code:`-a` tells the runner which action to perform.

In addition to :code:`--config` experiments, you can run :code:`--model` experiments.

These two cover fit/fit_transform and transform/predict, respectively.

The reason is that for fit/fit_tranform you typically require parameters, whereas
for transform/predict you start from a fitted model.

Continuing the example, we can transform the test data, using
the fitted model from before:

.. code-block:: shell

    $ smt run --model data/20170810-131658/RandomSelection.pkl -X data/X_test.npy -a transform
    >> Record label for this run: '20170810-134027'
    >> Data keys are [20170810-134027/X_new.npy(b33b0e0b794b64e5d284a602f5440620a21cac1c [2017-08-10 13:40:32])]

Again, Sumatra created an experiment record, which you can use to track input/output paths.

Writing your own models
-----------------------

.. _pipeline: ml_project/models/pipeline.py
.. _model_selection: ml_project/models/model_selection.py

In principle, the project framework can handle scikit-learn-style classes that implement
fit/fit_transform/transform/predict functions. In fact, it is recommended to
derive your estimator classes from the sklearn base classes, so that you can
take advantage of their extensive funcitonality.

This framework provides transformers_.


More tools
----------

.. _kaggle-cli: https://github.com/floydwch/kaggle-cli

A very convenient tool included in the ml-project framework is kaggle-cli_.
It can be used to submit predictions to kaggle and to view previous submissions.