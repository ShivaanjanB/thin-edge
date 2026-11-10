import sys

from .cli import main

# The guard matters on Windows: PyMC starts one worker process per chain, and each worker re-imports
# this module. Without it every worker would run the whole command again.
if __name__ == "__main__":
    sys.exit(main())
