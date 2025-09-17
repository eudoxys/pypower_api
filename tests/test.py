"""Run case tests"""
import os, sys
sys.path.append("../pypower_api")
from case import *
import numpy as np

test = Case("case9")
count,validation = test.validate()
if count > 0:
    print(validation)

assert test.N == 9, "incorrect number of busses"
assert test.L == 9, "incorrect number of branches"
assert test.G == 3, "incorrect number of generators"
assert test.D == 0, "incorrect number of dclines"

# test direct __getitem__ and __setitem__
bustypes = test["bus",np.s_[0:9],bus.BUS_TYPE].tolist()
test["bus",np.s_[0:9],bus.BUS_TYPE] = 1
assert (test["bus",np.s_[0:9],bus.BUS_TYPE] == np.ones((9,1))).all(), "__setitem__() failed"
test["bus",np.s_[0:9],bus.BUS_TYPE] = bustypes
assert (test["bus",np.s_[0:9],bus.BUS_TYPE] == np.array(bustypes)).all(), "__setitem() failed"

# test Data accessor
for n in range(1,test.N+1):
    assert test.Bus(str(n)).BUS_I == n, "invalid bus index"
try:
    test.Bus("1").BUS_I = 2
except AttributeError as err:
    assert str(err) == "property 'BUS_I' of 'Bus' object has no setter", "unexpected exception"
assert test.Bus("1").BUS_I == 1, "bus index changed unexpectedly"

assert test.Bus(0).BUS_TYPE == 3, "invalid bus type"
test.Bus(0).BUS_TYPE = 2
assert test.Bus(0).BUS_TYPE == 2, "bustype change failed"
test.Bus(0).BUS_TYPE = 3
assert test.Bus(0).BUS_TYPE == 3, "bustype reset failed"

# test Bus accessor objects
assert str(test.Bus(0)) == """{"BUS_I": 1.0, "BUS_TYPE": 3.0, "PD": 0.0, "QD": 0.0, "GS": 0.0, "BS": 0.0, "BUS_AREA": 1.0, "VM": 1.0, "VA": 0.0, "BASE_KV": 345.0, "ZONE": 1.0, "VMAX": 1.1, "VMIN": 0.9, "LAM_P": 0.0, "LAM_Q": 0.0, "MU_VMAX": 0.0, "MU_VMIN": 0.0}""", "Bus.str() failed"
assert repr(test.Bus(0)) == "Bus(case=Case('case9'),ref=0)", "Bus.repr() failed"

# test Branch accessor
assert test.Branch(0).F_BUS == 1, "branch F_BUS get failed"
assert test.Branch(0).T_BUS == 4, "branch T_BUS get failed"

# test Gen accessor
assert test.Gen(0).GEN_BUS == 1, "gen GEN_BUS get failed"
assert test.Gen("2").PG == 163.0, "gen PG get failed"
assert test.Gen(2).APF == 0.0, "gen APF get failed"

# test Gencost access
assert test.Gencost(0).MODEL == 2, "gencost MODEL get failed"
assert test.Gencost(0).NCOST == 3, "gencost N get failed"
assert (test.Gencost(0).COST == np.array([0.11,5,150])).all(), "gencost COST get failed"
test.Gencost(0).COST[1] = 200
assert test.Gencost(0).COST[1] == 200, "gencost COST set failed"
test.Gencost(0).COST = [0.11,5,150]
assert (test.Gencost(0).COST == np.array([0.11,5,150])).all(), "gencost COST reset failed"

# test iterators
for name in pp_index:
    for n,x in test.items(name,as_array=True):
        assert (getattr(test,name)[n] == x).all(), f"'{name}' iterator as array failed"
assert str(list(test.items("bus"))[0]) == "(0, Bus(case=Case('case9'),ref=0))", "bus iterator failed"

# test matrices
assert (test.matrix("A") == test.matrix("A",weighted=False)).all(), "A matrix unweighted failed"
A = test.matrix("A",weighted="complex")
assert (A.sum(axis=1).round(2)==[-17.36j,0,-17.06j,1.94-10.51j,1.28-5.59j,1.16-9.78j,1.62-13.7j,1.19-21.98j,1.37-11.6j ]).all(), "A matrix get complex failed"
assert (np.abs(A).round(2)<=test.matrix("A",weighted="abs").round(2)).all(), "A matrix get abs failed"
assert (test.matrix("A",weighted="real").sum(axis=1).round(2) == [0,0,0,58.82,25.64,84.03,117.65,31.25,100]).all(), "A matrix get real failed"
assert (test.matrix("A",weighted="imag").sum(axis=1).round(2) == [17.36,0,17.06,10.87,5.88,9.92,13.89,22.21,11.76]).all(), "A matrix get imag failed"

assert (test.matrix("D") == test.matrix("D",weighted=False)).all(), "D matrix unweighted failed"
D = test.matrix("D",weighted="complex")
assert (D.sum(axis=1).round(2)==[-17.36j,-16.j,-17.06j,3.31-39.48j,3.22-16.1j,2.44-32.44j,2.77-23.48j,2.8-35.67j,2.55-17.58j]).all(), "D matrix get complex failed"
assert (np.abs(D).round(2)<=test.matrix("D",weighted="abs").round(2)).all(), "D matrix get abs failed"
assert (test.matrix("D",weighted="real").sum(axis=1).round(2) == [0,0,0,158.82,84.46,109.67,201.68,148.9,131.25]).all(), "D matrix get real failed"
assert (test.matrix("D",weighted="imag").sum(axis=1).round(2) == [17.36,16,17.06,40,16.75,32.87,23.81,36.1,17.98]).all(), "D matrix get imag failed"

for weight in [None,False,True,"real","complex","abs","imag"]:
    assert ( test.matrix("L",weighted=weight) == test.matrix("D",weighted=weight) - test.matrix("A",weighted=weight) ).all(), f"L matrix get {weight} failed"

assert (test.matrix("E",weighted="real").round(1)==[  0.  +0.j ,   0.  +0.j ,   0.  +0.j ,  67.3 +0.j , 104.1+42.6j,
   104.1-42.6j, 172.8+41.7j, 172.8-41.7j, 213.6 +0.j ]).all(), "E matrix get real failed"

assert (test.matrix("V",weighted="complex")[0].round(2) == [ 0,1,0.51+0.16j,0.57,0.79,-0.51+0.24j,-0.52-0.04j,-0.23+0.03j,0.82]).all(), "V matrix failed"

assert (test.matrix("G") == [0,163,85,0,0,0,0,0,0]).all(), "G matrix get failed"
assert (test.matrix("S").round(1) == [0,0,0,0,94.9,0,105.9,0,134.6]).all(), "S matrix get failed"
assert (test.matrix("G") - test.matrix("S")).sum().round(1) == -87.4, "G - S matrix calculation failed"

for weight in [None,False,True,"real","complex","abs","imag"]:
    assert test.matrix('B',weighted=weight).sum().round(2) == 0.0, "B matrix get/sum failed"

assert test.matrix("Y").sum().round(2) == (17.1-215.17j), "Y matrix sum failed"
assert test.matrix("Z").sum().round(2) == (0.24+1.72j), "Z matrix sum failed"

b = test["bus",-1]
test.delete("bus",8)
test.add("bus",[9, 1, 125, 50, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9])
assert (b == test["bus",-1]).all(), "add/delete index failed"

b = test.Bus("9").to_dict()
test.delete("bus",8)
n = test.add("bus",**b)
b = test.Bus("9").to_array()
assert (b == test["bus",n]).all(), "add/delete ref failed"

# test solvers
for file in os.listdir("."):
    if re.match("case[0-9]+.py",file):
        test = Case(file,
            VERBOSE=0,
            OUT_ALL=0,
            OUT_SYS_SUM=False,
            OUT_AREA_SUM=False,
            OUT_BUS=False,
            OUT_GEN=False,
            OUT_ALL_LIM=0,
            OUT_V_LIM=0,
            OUT_LINE_LIM=0,
            OUT_PG_LIM=0,
            OUT_QG_LIM=0,
            )
        print(f"Testing {test.name}...",end="",flush=True,file=sys.stderr)
        count,report = test.validate()
        assert count == 0, f"{file}:\n# Validation failed\n\n{count} errors found.\n\n{'\n'.join([f'## Severity level {x}\n\n- {"\n- ".join(y)}' for x,y in report.items() if y])}"
        assert test.runpf(os.devnull,os.devnull), f"{file} runpf failed"
        assert test.rundcopf(os.devnull,os.devnull), f"{file} rundcopf failed"
        assert test.runduopf(os.devnull,os.devnull), f"{file} runduopf failed"
        assert test.runopf(os.devnull,os.devnull), f"{file} runopf failed"
        assert test.runuopf(os.devnull,os.devnull), f"{file} runuopf failed"
        sys.stdout.flush()
        print(file,"ok",file=sys.stderr,flush=True)

print("All tests ok",file=sys.stderr)