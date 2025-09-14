"""PyPOWER API main routine

Syntax: pypower_api [OPTIONS ... ] COMMAND [ARGUMENTS ...]

Commands:

  solve [pf|opf|dcopf|uopf|duopf]
  print [bus|branch|gen|gencost|dcline|dclinecost]

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

"""
import os
import sys
import argparse
import pandas as pd
try:
    from case import Case
    from index import pp_index
except ModuleNotFoundError:
    from .case import Case
    from .index import pp_index

EXEPATH = sys.argv[0]
EXEDIR = os.path.dirname(EXEPATH)
EXENAME = os.path.splitext(os.path.basename(EXEPATH))[0]
APPNAME = os.path.basename(EXEDIR)

E_OK = 0
E_SYNTAX = 1
E_FAILED = 2

def _solve(case,*args,**kwargs):

    try:
        if args[0] in ["pf","opf","dcopf","uopf","duopf"]:
            solver = getattr(case,"run"+args[0])
            del args[0]
        else:
            solver = getattr(case,"runpf")
        return solver(*args,**kwargs)
    except ModuleNotFoundError as err:
        raise FileNotFoundError(f"'{args[0]}' not found") from err

def _print(case,output=None,format="csv",*args,**kwargs):

    data = getattr(case,args[0])
    keys = pp_index[args[0]]
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
    if output is None and not result is None:
        print(result.strip())

def main():

    E_OK = 0
    E_SYNTAX = 1

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

    parser.add_argument("-i","--input",
        help="input file pathname",
        required=True,
        )
    parser.add_argument("-o","--output",
        help="output file pathname",
        default=None,
        )
    parser.add_argument("-f","--format",
        help="output data format",
        default="dataframe",
        )

    parser.add_argument("command",
        help="API command (use `pypower_api help` for details)",
        )
    parser.add_argument("arguments",
        nargs="?",
        help="API command arguments",
        )

    args = parser.parse_args()

    case = Case(args.input)

    if args.command == "help":

        print(__doc__)
        return(E_OK)

    if args.command == "solve":

        largs = (args.arguments 
            if isinstance(args.arguments,list) 
            else [args.arguments])
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
        largs = (args.arguments 
            if isinstance(args.arguments,list) 
            else [args.arguments])
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
