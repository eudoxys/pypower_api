# Copyright (c) 2025 Eudoxys Sciences LLC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Case accessor class"""

import os
import sys
import datetime as dt
import importlib
import io
import numpy as np
import re
import json
from typing import TypeVar, Any

import pypower
from pypower.ppoption import (ppoption,
    PF_OPTIONS, CPF_OPTIONS, OPF_OPTIONS, 
    OUTPUT_OPTIONS, PDIPM_OPTIONS, GUROBI_OPTIONS)
import pypower.runpf as pf
import pypower.rundcopf as dcopf
import pypower.runduopf as duopf
import pypower.runopf as opf
import pypower.runuopf as uopf
import pypower.runcpf as cpf

import pypower.idx_bus as bus
import pypower.idx_brch as branch
import pypower.idx_gen as gen
import pypower.idx_dcline as dcline
import pypower.idx_cost as cost

try:
    from index import pp_index
    from data import Data
    from bus import Bus
    from branch import Branch
    from gen import Gen
    from cost import Gencost, Dclinecost
    from dcline import Dcline
except ModuleNotFoundError:
    from .index import pp_index
    from .data import Data
    from .bus import Bus
    from .branch import Branch
    from .gen import Gen
    from .cost import Gencost, Dclinecost
    from .dcline import Dcline


DEBUG = True # enable raise on exceptions

def redirect_all(
        stdout:io.IOBase=sys.stdout,
        stderr:io.IOBase=sys.stderr,
        ):
    """Enables redirection of output from PP functions that import stdout and stderr from sys.
    """
    sys.path.append(os.path.dirname(pypower.__file__))
    for name in ["add_userfcn","dcopf_solver","hasPQcap","main",
            "makeAvl","makeBdc","makePTDF","makeYbus",
            "opf_args","opf_execute","opf_model","opf_setup",
            "pqcost","printpf",
            "qps_cplex","qps_gurobi","qps_ipopt","qps_mosek",
            "runcpf","runopf","runpf","runuopf",
            "savecase","scale_load",
            "toggle_iflims","toggle_reserves","total_load",
            ]:
        module = importlib.import_module(name)
        setattr(module,"stdout",stdout)
        setattr(module,"stderr",stderr)
        sys.stdout = stdout
        sys.stderr = stderr

class Case:
    """Case accessor class"""
    def __init__(self,
            case:dict|str=None,
            name:str="unnamed",
            **kwargs):
        """Case constructor"""
        self.args = dict(kwargs)

        self.name = name
        self.args["name"] = name

        self.version = "2"
        self.baseMVA = 100.0

        # initial empty case data
        for key,data in pp_index.items():
            setattr(self,key,np.zeros((0,len(data))))

        # load case data
        if not case is None:
            self.args["case"] = case
            self.read(case)

        self.index = {
            "bus" : [f"{x}" for x in self.bus[:,0]],
            "branch" : [f"{x}-{y}" for x,y in zip(self.branch[:,0],self.branch[:,1])],
            "gen" : [f"{x}" for x in self.gen[:,0]],
            "gencost" : [f"{x}" for x in self.gen[:,0]],
            "dcline" : [f"{x}-{y}" for x,y in zip(self.branch[:,0],self.dcline[:,1])] if hasattr(self,"dcline") else [],
            "dclinecost" : [f"{x}-{y}" for x,y in zip(self.branch[:,0],self.dcline[:,1])] if hasattr(self,"dcline") else [],
        }

        self.PF_OPTIONS = {x[0].upper():x[1] for x in PF_OPTIONS}
        self.CPF_OPTIONS = {x[0].upper():x[1] for x in CPF_OPTIONS}
        self.OPF_OPTIONS = {x[0].upper():x[1] for x in OPF_OPTIONS}
        self.OUTPUT_OPTIONS = {x[0].upper():x[1] for x in OUTPUT_OPTIONS}
        self.PDIPM_OPTIONS = {x[0].upper():x[1] for x in PDIPM_OPTIONS}
        self.GUROBI_OPTIONS = {x[0].upper():x[1] for x in GUROBI_OPTIONS}

        self.N = self.bus.shape[0] # number of busses
        self.L = self.branch.shape[0] # number of branches
        self.G = self.gen.shape[0] # number of generators
        self.D = self.dcline.shape[0] # number of dclines

        for key,arg in kwargs.items():

            if key in ["bus","branch","gen","gencost","dcline","dclinecost"]:

                setattr(self,key,np.array(arg))
                if key == "bus":
                    self.N = self.bus.shape[0]
                elif key == "branch":
                    self.L = self.branch.shape[0]
                elif key == "gen":
                    self.G = self.gen.shape[0]
                elif key == "dcline":
                    self.D = self.dcline.shape[0]

            else:
                self.set_options(**{key:arg})

        self._matrix = {}

    @property
    def case(self) -> dict:
        """Case property getter"""
        return {x:getattr(self,x) for x in ["version","baseMVA","bus","branch","gen",
                "gencost","dcline","dclinecost"] if hasattr(self,x) and not getattr(self,x) is None}
    @case.setter
    def case(self,case):
        """Case property setter"""
        for key,value in case.items():
            if not key in self.case:
                raise KeyError(f"{key=} is not a valid items in a case")
            dtype = type(self.case[key])
            if not isinstance(value,dtype):
                raise TypeError(f"{key=} dtype={type(value)} is invalid")
            self.case[key] = value
    
    @property
    def options(self) -> dict:
        """Options property getter"""
        return self.PF_OPTIONS | self.OPF_OPTIONS | self.CPF_OPTIONS \
            | self.OUTPUT_OPTIONS | self.PDIPM_OPTIONS | self.GUROBI_OPTIONS

    @options.setter
    def options(self,
            values:dict,
            ):
        """Options property setter"""
        for key,value in values.items():
            set_option(key,value)

    def __repr__(self):
        return f"{__class__.__name__}({repr(self.name)})"

    def set_options(self,
            **kwargs):
        for key,value in kwargs.items():
            found = False
            for options in [self.PF_OPTIONS,self.CPF_OPTIONS,self.OPF_OPTIONS,
                self.OUTPUT_OPTIONS,self.PDIPM_OPTIONS,self.GUROBI_OPTIONS]:
                if not key in options:
                    continue
                if not isinstance(value,type(options[key])):
                    raise TypeError(f"{key}={repr(value)} is invalid")
                options[key] = value
                found = True
                break
            if not found:
                raise ValueError(f"{key}={repr(value)} is invalid")

    def read(self,
            case:str|dict,
            ):
        """Read case from file or case data"""
        if isinstance(case,dict):
            for key,data in case.items():
                setattr(self,key,data)
        elif isinstance(case,str):
            sys.path.insert(0,os.path.dirname(case))
            self.name = os.path.splitext(os.path.basename(case))[0]
            module = importlib.import_module(self.name)
            call = getattr(module,self.name)
            for key,data in call().items():
                if key in pp_index and data.shape[1] < len(pp_index[key]):
                    # pad missing columns with zeros
                    data = np.concatenate([data,np.zeros((data.shape[0],len(pp_index[key])-data.shape[1]))],axis=1)
                setattr(self,key,data)
            sys.path = sys.path[1:]
        else:
            raise ValueError("case must be either case data or a case filename")

    def to_dict(self) -> dict:
        """Convert case to dict"""
        return self.case

    def write_py(self,
            file:str=None,
            ):
        """Write case to file"""
        file = file if file else self.name
        if not file.endswith(".py"):
            file += ".py"
        name = os.path.splitext(os.path.basename(file))[0]
        with open(file,"w") as fh:
            print(f"# Generated by '{repr(self)}.write({file=})' at {dt.datetime.now()}",file=fh)
            print(f"from numpy import array, float64",file=fh)
            print(f"def {name}():",file=fh)
            print(f"   return {{",file=fh)
            for tag,data in self.case.items():
                if hasattr(data,"tolist"):
                    data = data.tolist() # change np.array to list
                if isinstance(data,list):
                    print(f"""    "{tag}": array([""",file=fh)
                    if tag in pp_index:
                        print("      #",",".join([f"{x:>10.10s}" for x in pp_index[tag]]),file=fh)
                    for row in data:
                        print(f"""      [ {','.join([f'{x:10.5g}' for x in row])}],""",file=fh)
                    print("    ],dtype=float64),",file=fh)
                else:
                    print(f"""    "{tag}": {repr(data)},""",file=fh)
            print(f"}}",file=fh)

    def write_csv(self,
            file:str=None,
            ):
        """Write CSV file"""
        with open(file,"w") as fh:
            print("application,pypower",file=fh)
            print(f"version,{self.version}",file=fh)
            print(f"baseMVA,{self.baseMVA}",file=fh)
            for key,data in [(x,y) for x,y in self.case.items() if isinstance(y,np.ndarray) and len(y) > 0]:
                print(",".join([key,str(len(data))]),file=fh)
                print(",".join(["ID"]+pp_index[key]),file=fh)
                for n,row in enumerate(data.tolist()):
                    print(",".join([str(n)]+[str(x) for x in row]),file=fh)

    def write_json(self,
            file:str=None,
            ):
        """Write JSON file"""
        with open(file,"w") as fh:
            result = {
                "application" : "pypower",
                "version" : self.version,
                "basemva" : self.baseMVA,
            }
            for key,value in [(x,y) for x,y in self.case.items() if isinstance(y,np.ndarray)]:
                result[key] = self.case[key].tolist()
            json.dump(result,fh,indent=2)

    def write(self,
            file:str=None,
            ):
        """Save case to file format"""
        ext = os.path.splitext(file)[1]
        if ext == ".py":
            return self.write_py(file)
        if ext == ".csv":
            return self.write_csv(file)
        if ext == ".json":
            return self.write_json(file)
        raise ValueError(f"{ext=} is not a supported file format")

    def run(self,
            call:callable,
            *args,
            stdout=sys.stdout,
            stderr=sys.stderr,
            **kwargs) -> bool:
        """Run PP solver"""
        if isinstance(stdout,str):
            stdout = open(stdout,"w")
        if not isinstance(stdout,io.IOBase):
            raise TypeError("stdout is not an IO stream")
        if isinstance(stderr,str):
            stderr = open(stderr,"w")
        if not isinstance(stderr,io.IOBase):
            raise TypeError("stderr is not an IO stream")
        self.set_options(**kwargs)
        try:
            redirect_all(stdout,stderr)
            result = call(self.case,*args,ppopt=self.options)
            if isinstance(result,tuple):
                result,status = result
            else:
                status = result["success"]
            error = None
        except Exception as err:
            if DEBUG:
                raise
            result,status = None,0
            error = sys.exc_info()
        self.result = {
            "data" : result,
            "status" : status,
            "error" : error,
        }
        sys.stdout.flush()
        sys.stderr.flush()
        redirect_all()
        return status == 1

    def runpf(self,
            stdout:io.IOBase|str=sys.stdout,
            stderr:io.IOBase|str=sys.stderr,
            **kwargs,
            ) -> bool:
        """Run powerflow solver"""
        return self.run(pf.runpf,stdout=stdout,stderr=stderr)

    def rundcopf(self,
            stdout:io.IOBase|str=sys.stdout,
            stderr:io.IOBase|str=sys.stderr,
            **kwargs,
            ):
        """Run DC OPF solver"""
        return self.run(dcopf.rundcopf,stdout=stdout,stderr=stderr)

    def runduopf(self,
            stdout:io.IOBase|str=sys.stdout,
            stderr:io.IOBase|str=sys.stderr,
            **kwargs,
            ):
        """Run DC unit-commitment OPF solver"""
        return self.run(duopf.runduopf,stdout=stdout,stderr=stderr)

    def runopf(self,
            stdout:io.IOBase|str=sys.stdout,
            stderr:io.IOBase|str=sys.stderr,
            **kwargs,
            ):
        return self.run(opf.runopf,stdout=stdout,stderr=stderr)

    def runuopf(self,
            stdout:io.IOBase|str=sys.stdout,
            stderr:io.IOBase|str=sys.stderr,
            **kwargs,
            ):
        """Run unit-commitment OPF"""
        return self.run(uopf.runuopf,stdout=stdout,stderr=stderr)

    def runcpf(self,
            target:dict,
            stdout:io.IOBase|str=sys.stdout,
            stderr:io.IOBase|str=sys.stderr,
            **kwargs,
            ):
        """Run continuation powerflow"""
        return self.run(cpf.runcpf,target,stdout=stdout,stderr=stderr)

    def __getitem__(self,
            ref:str|tuple|list,
            ) -> int|float|np.float64:
        """Get an object data item"""
        if isinstance(ref,str):
            return getattr(self,ref)
        elif len(ref) == 1:
            return getattr(self,ref[0])
        elif len(ref) == 2:
            return getattr(self,ref[0])[ref[1]]
        elif len(ref) == 3:
            return getattr(self,ref[0])[ref[1],ref[2]]
        else:
            raise KeyError(f"{ref=} is not valid")

    def __setitem__(self,
            ref:tuple|list,
            value:int|float|np.float64,
            ):
        """Set an object data item"""
        dtype = getattr(self,ref[0]).dtype
        getattr(self,ref[0])[ref[1],ref[2]] = np.float64(value)
        self._matrix = {} # reset matrix results

    def Bus(self,name:int|str):
        """Get a bus object"""
        return Bus(self,name)

    def Branch(self,name:int|str):
        """Get a branch object"""
        return Branch(self,name)

    def Gen(self,name:int|str):
        """Get a generator object"""
        return Gen(self,name)

    def Gencost(self,name:int|str):
        """Get a generator cost object"""
        return Gencost(self,name)

    def Dcline(self,name:int|str):
        """Get a DC line object"""
        return Dcline(self,name)

    def Dclinecost(self,name:int|str):
        """Get a DC line cost object"""
        return Dclinecost(self,name)

    def add(self,ref:str,values:list|TypeVar('np.array')=None,**kwargs):
        """Add an object to the case"""
        if ref not in pp_index:
            raise ValueError(f"{ref=} is not a valid object type")

        data = getattr(self,ref)
        row = np.zeros(shape=(1,data.shape[1]))
        n = data.shape[0]
        if not values is None:
            row[0,0:len(values)] = values
        ndx = {x:n for n,x in enumerate(pp_index[ref])}
        for key,value in kwargs.items():
            if not key in ndx:
                raise KeyError(f"{key}={value} is not a valid property of {ref} objects")
            row[0,ndx[key]] = value
        setattr(self,ref,np.vstack([data,row]))

        self.N = self.bus.shape[0] # number of busses
        self.L = self.branch.shape[0] # number of branches
        self.G = self.gen.shape[0] # number of generators
        self.D = self.dcline.shape[0] # number of dclines

        return n

    def delete(self,name:str,ref:str|int):
        """Delete an object from the case"""
        if name not in pp_index:
            raise f"{name=} is not valid"

        self.N = self.bus.shape[0] # number of busses
        self.L = self.branch.shape[0] # number of branches
        self.G = self.gen.shape[0] # number of generators
        self.D = self.dcline.shape[0] # number of dclines

        setattr(self,name,np.delete(getattr(self,name),ref,axis=0))

    def items(self,
            name:str,
            as_array:bool=False,
            ) -> tuple[int,TypeVar('np.array')|TypeVar('Data')]:
        """Iterate through items in a case"""
        if not name in pp_index:
            raise KeyError(f"'{name=}' is not valid")
        for n,x in enumerate(getattr(self,name)):
            yield n,x if as_array else getattr(self,name.title())(n)

    def matrix(self,
            name:str,
            sparse:bool=None,
            weighted:bool|str|None=None,
            dtype:Any=None,
            **kwargs) -> TypeVar('np.array'):
        """Get a graph analysis matrix"""
        cache = str(f"{name}-{sparse}-{weighted}-{dtype}")
        if cache in self._matrix: # cached result

            return self._matrix[cache]

        # A used is multiple places, almost always needed
        A_cache = "-".join(["A"]+cache.split("-")[1:])
        if A_cache in self._matrix:
            A = self._matrix[A_cache]
        else:
            A = np.zeros((self.N,self.N),dtype=complex if weighted=="complex" else float)
            for n,x in [(n,x) for n,x in self.items("branch",as_array=True) if x[branch.BR_STATUS] == 1]:
                i,j = int(x[branch.F_BUS])-1,int(x[branch.T_BUS])-1
                if weighted in [None,False]:
                    Y = 1
                elif weighted in ["abs",True]:
                    Z = complex(x[branch.BR_R],x[branch.BR_X])
                    Y = abs(1/Z) if abs(Z)>0 else 0.0
                elif weighted in ["real"]:
                    Z = x[branch.BR_R]
                    Y = 1/Z if abs(Z)>0 else 0.0
                elif weighted in ["imag"]:
                    Z = x[branch.BR_X]
                    Y = 1/Z if abs(Z)>0 else 0.0
                elif weighted == "complex":
                    Z = complex(x[branch.BR_R],x[branch.BR_X])
                    Y = 1/Z if abs(Z)>0 else 0.0
                else:
                    raise ValueError(f"{weighted=} is invalid")
                A[i,j] += Y
            self._matrix[A_cache] = A

        # D used is multiple places, almost always needed
        D_cache = "-".join(["D"]+cache.split("-")[1:])
        if D_cache in self._matrix:
            D = self._matrix[D_cache]
        else:
            D = np.zeros((self.N,self.N),dtype=complex if weighted=="complex" else float)
            for n,x in [(n,x) for n,x in self.items("branch",as_array=True) if x[branch.BR_STATUS] == 1]:
                i,j = int(x[branch.F_BUS])-1,int(x[branch.T_BUS])-1
                if weighted in [None,False]:
                    Y = 1
                elif weighted in ["abs",True]:
                    Z = complex(x[branch.BR_R],x[branch.BR_X])
                    Y = abs(1/Z) if abs(Z)>0 else 0.0
                elif weighted in ["real"]:
                    Z = x[branch.BR_R]
                    Y = 1/Z if abs(Z)>0 else 0.0
                elif weighted in ["imag"]:
                    Z = x[branch.BR_X]
                    Y = 1/Z if abs(Z)>0 else 0.0
                elif weighted == "complex":
                    Z = complex(x[branch.BR_R],x[branch.BR_X])
                    Y = 1/Z if abs(Z)>0 else 0.0
                else:
                    raise ValueError(f"{weighted=} is invalid")
                D[i,i] += Y
                D[j,j] += Y
            self._matrix[D_cache] = D

        E_cache = "-".join(["E"]+cache.split("-")[1:])
        V_cache = "-".join(["V"]+cache.split("-")[1:])

        if name == "A": # adjacency

            result = self._matrix[A_cache]

        elif name == "B": # incidence

            B = np.zeros(shape=(self.N,self.L),dtype=complex if weighted=="complex" else float)
            for n,x in [(n,x) for n,x in self.items("branch",as_array=True) if x[branch.BR_STATUS] == 1]:
                i,j = int(x[branch.F_BUS])-1,int(x[branch.T_BUS])-1
                if weighted in [None,False]:
                    B[i,n] = -1
                    B[j,n] = 1
                elif weighted == True:
                    B[i,n] += -1
                    B[j,n] += 1
                elif weighted == "real":
                    Z = x[branch.BR_R]
                    Y = 1/Z if abs(Z) > 0 else 0.0
                    B[i,n] += -Y
                    B[j,n] += Y
                elif weighted == "imag":
                    Z = x[branch.BR_X]
                    Y = 1/Z if abs(Z) > 0 else 0.0
                    B[i,n] += -Y
                    B[j,n] += Y
                elif weighted == "complex":
                    Z = complex(x[branch.BR_R],x[branch.BR_X])
                    Y = 1/Z if abs(Z) > 0 else 0.0
                    B[i,n] += -Y
                    B[j,n] += Y
                elif weighted == "abs":
                    Z = complex(x[branch.BR_R],x[branch.BR_X])
                    Y = 1/Z if abs(Z) > 0 else 0.0
                    B[i,n] += -abs(Y)
                    B[j,n] += abs(Y)
                else:
                    raise ValueError(f"{weighted=} is invalid")

            result = B

        elif name == "D": # degree

            result = self._matrix[D_cache]

        elif name == "E": # eigenvalues of L

            if E_cache in self._matrix:
                result = self._matrix[E_cache]
            e,v = np.linalg.eig(self.matrix("L",weighted=weighted))
            n = [n for x,n in sorted([(x,n) for n,x in enumerate(np.abs(e) if e.dtype == complex else e)])]
            result = e[n],v[n]
            self._matrix[E_cache],self._matrix[V_cache] = result
            result = self._matrix[E_cache]

        elif name == "G": # generation

            G = np.zeros(shape=(self.N),dtype=dtype)
            for n,x in self.items("gen",as_array=True):
                g = complex(x[gen.PG],x[gen.QG]) if x[gen.GEN_STATUS] == 1 else 0
                G[n] += g if dtype == complex else np.abs(g)
            result = G

        elif name == "Gmin": # generation min

            G = np.zeros(shape=(self.N),dtype=dtype)
            for n,x in self.items("gen",as_array=True):
                g = complex(x[gen.PMIN],x[gen.QMIN]) if x[gen.GEN_STATUS] == 1 else 0
                G[n] += g if dtype == complex else np.abs(g)
            result = G

        elif name == "Gmax": # generation max

            G = np.zeros(shape=(self.N),dtype=dtype)
            for n,x in self.items("gen",as_array=True):
                g = complex(x[gen.PMAX],x[gen.QMAX]) if x[gen.GEN_STATUS] == 1 else 0
                G[n] += g if dtype == complex else np.abs(g)
            result = G

        elif name == "L": # Laplacian

            result = self._matrix[D_cache] - self._matrix[A_cache]

        elif name == "S": # demand

            S = np.zeros(shape=(self.N),dtype=dtype)
            for n,x in self.items("bus",as_array=True):
                s = complex(x[bus.PD],x[bus.QD])
                S[n] += s if dtype == complex else np.abs(s)
            result = S

        elif name == "V": # eigenvectors of L

            if V_cache in self._matrix:
                result = self._matrix[V_cache]
            e,v = np.linalg.eig(self.matrix("L",weighted=weighted))
            n = [n for x,n in sorted([(x,n) for n,x in enumerate(np.abs(e) if e.dtype == complex else e)])]
            result = e[n],v[n]
            self._matrix[E_cache],self._matrix[V_cache] = result
            result = self._matrix[V_cache]

        elif name == "Y": # admittance

            Y = np.zeros(shape=(self.N,self.N),dtype=complex)
            for n,x in [(n,x) for n,x in self.items("branch",as_array=True) if x[branch.BR_STATUS] == 1]:
                i,j = int(x[branch.F_BUS])-1,int(x[branch.T_BUS])-1
                Z = complex(x[branch.BR_R],x[branch.BR_X])
                Y[i,j] += 1/Z if abs(Z) > 0 else 0.0
                Y[j,i] += 1/Z if abs(Z) > 0 else 0.0
                f,t = self["bus",i],self["bus",j]
                Y[i,i] += complex(f[bus.GS],f[bus.BS])
                Y[j,j] += complex(t[bus.GS],t[bus.BS])
            result = Y

        elif name == "Z": # impedance

            # TODO: need a better way to compute Z that doesn't warn 1/0
            result = np.nan_to_num(1/self.matrix("Y"),nan=0.0,posinf=0.0,neginf=0.0)

        else:

            raise KeyError(f"matrix '{name=}' is invalid")

        self._matrix[cache] = result
        return result

    def validate(self) -> bool:
        """Validate a case"""

        result = {
            # severity levels
            0 : [], # unusual condition, failure unlikely
            1 : [], # unreasonable condition, failure possible
            2 : [], # improbable condition, failure likely
            3 : [], # invalid condition, failure certain
        }

        # check bus values
        count = 0
        for n,x in self.items("bus"):
            if x.BUS_I <= 0 or x.BUS_I > self.N:
                result[3].append(f"bus {n} has invalid BUS_I={x.BUS_I}")
                count += 1
            # TODO
        return count,result
