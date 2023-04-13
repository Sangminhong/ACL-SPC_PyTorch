import readline
import rlcompleter
readline.parse_and_bind("tab: complete")
import code
import pdb

import torch.multiprocessing as mp

# debugging tools
def interact(local=None):
    """interactive console with autocomplete function. Useful for debugging.
    interact(locals())
    """
    if local is None:
        local=dict(globals(), **locals())

    readline.set_completer(rlcompleter.Completer(local).complete)
    code.interact(local=local)

def set_trace(local=None):
    """debugging with pdb
    """
    if local is None:
        local=dict(globals(), **locals())

    pdb.Pdb.complete = rlcompleter.Completer(local).complete
    pdb.set_trace()