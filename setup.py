"""Setup script for ml-project"""
import sys
import subprocess
from os.path import normpath


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
    elif sys.version_info.minor < 5:
        action = getattr(subprocess, "call")
    else:
        action = getattr(subprocess, "run")

    action(["conda", "env", "create", "-n", "ml_project", "-f",
            ".environment"])

    action(["bash", "-c", "source activate ml_project && "
            "smt init -d {datapath} -i {datapath} -e python -m run.py "
            "-c error -l cmdline ml_project".format(
                datapath=normpath('./data'))])

    print("\n========================================================")
    print("Type 'source activate ml_project' to activate environment.")
    print("==========================================================")


if __name__ == '__main__':
    setup()
