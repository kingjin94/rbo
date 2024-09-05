from pathlib import Path

this_file = Path(__file__)
MODULE = this_file.parent.parent
SRC = MODULE.parent
ROOT = SRC.parent
TESTS = ROOT.joinpath('tests')
DATA = ROOT.joinpath('data')
