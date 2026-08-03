"""Feature generation: protein language model embeddings and literature TF-IDF.

These steps are run once to produce the matrices that ``protgoat.train`` consumes.
The outputs for CAFA5 are published, so reproducing a training run does not
require re-running them -- see ``docs/data.md``.
"""

from . import abstracts, plm

__all__ = ["abstracts", "plm"]
