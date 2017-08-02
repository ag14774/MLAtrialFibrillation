import sys
import subprocess

if sys.version_info.major < 3:
    action = getattr(subprocess, "call")
else:
    action = getattr(subprocess, "run")

action(["conda", "env", "create", "-n", "ml-project", "-f", "environment.yml"])
#action(["mkdir", "/tmp/smt"])
action(["bash", "-c", "source activate ml-project && "
        "smt init -d ./archive -i ./archive -e python -m run.py "
        "-c store-diff ml-project"])