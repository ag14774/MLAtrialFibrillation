ETH Machine Learning Projects
=============================

.. _scikit-learn: http://scikit-learn.org/stable/
.. _sklearn-dev-guide: http://scikit-learn.org/stable/developers/index.html
.. _sumatra: https://pythonhosted.org/Sumatra/
.. _miniconda: https://conda.io/docs/install/quick.html
.. _pipeline: ml_project/pipeline.py
.. _gridsearch: ml_project/model_selection.py
.. _`example config`: .example_config.yaml
.. _VirtualBox: https://www.virtualbox.org/
.. _Ubuntu: https://www.ubuntu.com/download/desktop
.. _data: data/
.. _kaggle-cli: https://github.com/floydwch/kaggle-cli
.. _kaggle: https://inclass.kaggle.com/c/ml-project-1

This repository contains the framework for the practical projects offered
during the *Machine Learning* course at ETH Zurich. It serves two main purposes:

* Convenient execution of machine learning models conforming to the scikit-learn_ pattern.
* Structured & reproducible experiments by integration of sumatra_ and miniconda_.

Getting Started (Non-Linux)
---------------------------

The project framework has been tested in Linux (Ubuntu) environments only. If you
are using Linux already, you can skip forward to Getting Started for Linux.

If you are using Mac or Windows, please install VirtualBox_ and create an 64-bit Ubuntu_
virtual machine (VM).

Make sure you allocate sufficient RAM (>= 8GB) and disk space (>= 64GB) for the VM.

If you can not choose 64-bit Ubuntu in VirtualBox, you might have to enable
virtualization in your BIOS.

Once your VM is running, you only need to install git:

.. code-block:: shell

    sudo aptitude install git

(Code blocks in this README show Unix shell commands.)

After that, please continue with Getting Started for Linux.

Getting Started (Linux)
-----------------------

First you need to install miniconda_ on your system.

Having installed miniconda, clone the repository and run the setup script:

.. code-block:: shell

    git clone https://gitlab.vis.ethz.ch/vwegmayr/ml-project.git
    cd ml-project
    python setup.py

Make sure you have downloaded the data to the data_ folder, either by using the
kaggle-cli_ tool or from the kaggle_ homepage.

Running an example experiment
-----------------------------

Make sure the environment is activated:

.. code-block:: shell

    source activate ml_project

If you encounter problems with site-packages try:

.. code-block:: shell

    export PYTHONNOUSERSITE=True; source activate ml_project

To run an example experiment, simply type

.. code-block:: shell

    smt run --config .example_config.yaml -X data/X_train.npy -a fit_transform

    >> =========== Config ===========
    >> {'class': <class 'ml_project.models.transformers.RandomSelection'>,
    >> 'params': {'n_components': 1000, 'random_state': 37}}
    >> ==============================

    >> Record label for this run: '20170810-131658'
    >> Data keys are [20170810-131658/RandomSelection.pkl(9b028327c83a153c0824ca8701f3b78a5106071c [2017-08-10 13:17:04]),
    >> 20170810-131658/X_new.npy(b8c093d7c8e13399b6fe4145f14b4dbc0f241503 [2017-08-10 13:17:04])]

The default experiment will reduce the dimensionality of the training data by
selecting 1000 dimensions at random.

Results can be found in timestamped directories :code:`data/YYYYMMDD-hhmmss`, i.e. for the experiment shown above, you would find the results in
:code:`data/20170810-131658`.

It produced two outputs, first the fitted model *RandomSelection.pkl* and second
the transformed training data *X_new.npy*.

To view the experiment record, type :code:`smtweb`.

This command will open a new window in your webbrowser, where you can explore
the information stored about the example experiment.

You can choose from different examples in the `example config`_ file.

More details on experiments
---------------------------

Let us consider the above command in more detail:

.. code-block:: shell

    smt run --config .example_config.yaml -X data/X_train.npy -a fit_transform

* :code:`smt` invokes sumatra_, which is an experiment tracking tool.

* :code:`run` tells sumatra_ to execute the experiment runner.

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

    smt run --model data/20170810-131658/RandomSelection.pkl -X data/X_test.npy -a transform
    >> Record label for this run: '20170810-134027'
    >> Data keys are [20170810-134027/X_new.npy(b33b0e0b794b64e5d284a602f5440620a21cac1c [2017-08-10 13:40:32])]

Again, sumatra_ created an experiment record, which you can use to track input/output paths.

Writing your own models
-----------------------

The project framework can handle sklearn-style classes that implement
fit/fit_transform/transform/predict functions.

Please implement your models as classes which conform with the sklearn pattern.
With this common strucutre, you can easily read and reuse code created by others.

In general, it is recommended to take advantage of the extensive functionality of the sklearn API.

Make sure to read the sklearn-dev-guide_, especially the sections *Coding guidelines*,
*APIs of scikit-learn objects*, and *Rolling your own estimator*.

Furthermore, take advantage that sklearn is open source.

Look at their code, it is very instructive!

This framework already implements an interface to the sklearn classes pipeline_
and gridsearch_.

Check out the `example config`_ to find out more about how to use them.

Code Submission
---------------

It is required to publish your code shortly after the kaggle submission deadline
(kaggle submission deadline + 24 hours).

First, you have to make sure that your code passes the flake8 tests.
You can check by running

.. code-block:: shell

    flake8

in the ml-project folder. It will return a list of coding quality errors.

Try to run it every now end then, otherwise the list of fixes you have to do before submission may get rather long.

Next, create and push a new branch which is named :code:`legi-number/ml-project-1`, e.g.

.. code-block:: shell
    git checkout -b 17-123-456/ml-project-1
    git push origin 17-123-456/ml-project-1

The first part has to be your Legi-Number, the number in the second part identifies the project.

This repository runs an automatic quality check, when you push your branch.
Additionally, the timestamp of the push is checked.

Results are only accepted, if the checks are positive and submission is before the deadline.

.. figure:: https://gmkr.io/s/5995a0c7022cf3566f9c65c5/0

    Check under *Pipelines*, if your commit passed the check.
    The *latest* flag indicates which commit is the most current.

More tools
----------

A very convenient tool included in the ml-project framework is kaggle-cli_.
It can be used to submit predictions to kaggle and to view previous submissions.
