"""PyPOWER API main routine

Syntax: pypower_api [OPTIONS ... ] COMMAND [ARGUMENTS ...]

Commands:

  matrix [A|B|D|E|G|Gmin|Gmax|L|S|V|W|Y|Z]

    Prints the graph matrix as follows:

        A - adjacency
        B - incidence
        D - degree
        E - eigenvalues of L
        G - generation
        Gmin - minimum generation
        Gmax - maximum generation
        L - Laplacian
        S - loads
        V - eigenvectors of L
        Y - admittance
        Z - impedance

  print [bus|branch|gen|gencost|dcline|dclinecost]

    Prints the model component

  solve [pf|opf|dcopf|uopf|duopf]

    Solver the network problem

  version

    Displays the current API version

Options:

  -h, --help            show this help message and exit
  -v, --verbose         enable verbose output
  -q, --quiet           disable non-error output
  -s, --silent          disable all output
  -d, --debug           enable exception traceback
  -w, --warning         disable warning output
  -i INPUT, --input INPUT
                        input file pathname
  -o OUTPUT, --output OUTPUT
                        output file pathname
  -f FORMAT, --format FORMAT
                        output data format
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import importlib.metadata as meta

# API modules
try:
    from case import Case
    from index import pp_index as _index
except ModuleNotFoundError:
    from .case import Case
    from .index import pp_index as _index

# executable path/name
APPNAME = "pypower_api"
try:
    VERSION = meta.version(APPNAME)
except:
    VERSION = "0.0.0-dev"
EXEPATH = sys.argv[0]
EXEDIR = os.path.dirname(EXEPATH)
EXENAME = os.path.splitext(os.path.basename(EXEPATH))[0]

np.set_printoptions(edgeitems=30000,linewidth=30000)

# exit codes
E_OK = 0
E_SYNTAX = 1
E_FAILED = 2

def _solve(case:Case,*args,**kwargs):
    """Network solvers"""
    try:
        if args[0] in ["pf","opf","dcopf","uopf","duopf"]:
            solver = getattr(case,"run"+args[0])
            del args[0]
        else:
            solver = getattr(case,"runpf")
        return solver(*args,**kwargs)
    except ModuleNotFoundError as err:
        raise FileNotFoundError(f"'{args[0]}' not found") from err

def _print(case:Case,
        output:str|None=None,
        format:str="csv",
        *args,**kwargs):
    """Network print"""
    data = getattr(case,args[0])
    keys = _index[args[0]]
    df = pd.DataFrame(dict(zip(keys,data.T)))
    df.index.name="ID"
    if len(df) == 0:
        return
    if format == "csv":
        result = df.to_csv(output,**kwargs)
    elif format == "json":
        result = df.T.to_json(output,**kwargs)
    elif format == "dataframe":
        print(df,file=output)
        result = None
    else:
        raise ValueError(f"{format=} is invalid")
    if output is None and not result is None:
        print(result.strip())

def _matrix(case,
        output=None,
        format="csv",
        formatter=str,
        *args,**kwargs):
    """Network analysis"""
    data = case.matrix(*args,**kwargs)
    if isinstance(output,str):
        output = open(output,"w")
    elif output is None:
        output = sys.stdout
    if format == "csv":
        for row in data:
            print(",".join([formatter(x) for x in row]),file=output)
    else:
        raise ValueError(f"{format=} is invalid")

def main() -> int:
    """Main routine"""

    parser = argparse.ArgumentParser(
        prog=APPNAME,
        description="PyPOWER API",
        epilog="See https://github.com/eudoxys/pypower_api"\
            " for more information",
        )

    parser.add_argument("-v","--verbose",
        action="store_true",
        help="enable verbose output",
        )
    parser.add_argument("-q","--quiet",
        action="store_true",
        help="disable non-error output",
        )
    parser.add_argument("-s","--silent",
        action="store_true",
        help="disable all output",
        )
    parser.add_argument("-d","--debug",
        action="store_true",
        help="enable exception traceback",
        )
    parser.add_argument("-w","--warning",
        action="store_false",
        help="disable warning output",
        )
    parser.add_argument("--version",
        action="store_true",
        help="display the version information",
        )

    parser.add_argument("-i","--input",
        help="input file pathname",
        default=None
        )
    parser.add_argument("-o","--output",
        help="output file pathname",
        default=None,
        )
    parser.add_argument("-f","--format",
        help="output data format",
        default="csv",
        )

    parser.add_argument("command",
        help="API command (use `pypower_api help` for details)",
        nargs="*",
        )

    args = parser.parse_args()

    if args.version or args.command[0] == "version":
        print(VERSION)
        return E_OK

    if args.command[0] == "help":

        print(__doc__)
        return(E_OK)

    if args.input is None:

        print(f"ERROR [{APPNAME}]: missing input")

    case = Case(args.input)

    largs = [] if len(args.command) < 2 else args.command[1:]

    if args.command[0] == "solve":

        pargs = [x for x in largs if not "=" in x]
        kargs = {x[0]:"=".join(x[1:]) for x in largs if "=" in x}
        try:
            result = _solve(case,*pargs,**kargs)
        except:
            e_type, e_name, _ = sys.exc_info()
            print(f"ERROR [{APPNAME}]: {e_type.__name__} - {e_name}")
            if args.debug:
                raise
            return E_FAILED
        return E_OK if result else E_FAILED

    if args.command == "print":

        pargs = [x for x in largs if not "=" in x]
        kargs = {x[0]:"=".join(x[1:]) for x in largs if "=" in x}
        try:
            _print(case,args.output,args.format,*pargs,**kargs)
        except:
            e_type, e_name, _ = sys.exc_info()
            print(f"ERROR [{APPNAME}]: {e_type.__name__} - {e_name}")
            if args.debug:
                raise
            return E_FAILED
        return E_OK

    if args.command == "matrix":

        pargs = [x for x in largs if not "=" in x]
        kargs = {x[0]:"=".join(x[1:]) for x in largs if "=" in x}
        try:
            _matrix(case,args.output,args.format,str,*pargs,**kargs)
        except:
            e_type, e_name, _ = sys.exc_info()
            print(f"ERROR [{APPNAME}]: {e_type.__name__} - {e_name}")
            if args.debug:
                raise
            return E_FAILED
        return E_OK


    print(f"ERROR [{APPNAME}]: '{args.command}' is invalid",file=sys.stderr)
    return E_FAILED

if __name__ == "__main__":

    print(__doc__.strip())

    print("\n*** GLOBALS ***\n")
    for name,value in [(x,globals()[x]) 
            for x in globals() 
            if not x.startswith("_") 
                and not isinstance(globals()[x],type(os)) 
                and not callable(globals()[x])]:
        print(f"  - {name}={value}")

    print("\n*** CALLABLES ***\n")
    for name,value in [(x,globals()[x]) 
            for x in globals() 
            if not x.startswith("_") 
                and not isinstance(globals()[x],type(os)) 
                and callable(globals()[x])
                and getattr(globals()[x],"__doc__")]:
        print(f" - {name}: {value.__doc__}")
