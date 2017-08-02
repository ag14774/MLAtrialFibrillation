"""Setup script for ml-project"""
import sys
import subprocess


def setup():
    """Setup function

    Requires:
        Installation of `miniconda`_.

    Todo:
        * Include automatic installtion of miniconda.

    .. _miniconda:
       https://conda.io/docs/install/quick.html#linux-miniconda-install

    """
    if sys.version_info.major < 3:
        action = getattr(subprocess, "call")
    else:
        action = getattr(subprocess, "run")

    action(["conda", "env", "create", "-n", "ml_project", "-f", ".environment"])

    action(["bash", "-c", "source activate ml_project && "
            "smt init -d ./data -i ./data -e python -m run.py "
            "-c store-diff -l cmdline ml_project"])
    print("Please type 'source activate ml_project' to activate environment.")


if __name__ == '__main__':
    setup()