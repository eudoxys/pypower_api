# PyPOWER API

A Python convenience API is available for developers who wish to use PyPOWER as part of their applications.  The API provides the following convenience classes

The `Case`: handles cases, include creating, reading, editing, solving, and writing cases. Cases use the following data accessor classes:

* `Bus`: access bus objects

* `Branch`: access branch objects

* `Gen`: access generator objects

* `Gencost`: access generator cost objects

* `Dcline`: access DC line objects

* `Dclinecost`: access DC line cost objects

The `Case` class also provide runners for powerflow, OPF, and CDF solvers, including non-linear heuristic solvers.  In addition, all solver options may be set using the `Case` class.

## Create a new case

## Adding objects

## Delete objects

## Access object data

## Iterating over objects

## Saving a case

## Reading a case

## Solving a case

## Extracting matrices

# Command Line

Syntax: `pypower_api [OPTIONS ... ] COMMAND [ARGUMENTS ...]`

## Commands

  Print the graph matrix

    matrix [A|B|D|E|G|Gmin|Gmax|L|S|V|W|Y|Z]

  * `A` - adjacency
  * `B` - incidence
  * `D` - degree
  * `E` - eigenvalues of $L$
  * `G` - generation
  * `Gmin` - minimum generation
  * `Gmax` - maximum generation
  * `L` - Laplacian
  * `S` - loads
  * `V` - eigenvectors of $L$
  * `Y` - admittance
  * `Z` - impedance

  Print the model components

    print [bus|branch|gen|gencost|dcline|dclinecost]

  Save model in JSON, CSV, or PY format

    save [-o] FILENAME.[py,csv,json]

  Solve the network problem

    solve [pf|opf|dcopf|uopf|duopf]

  Displays the current API version

    version

## Options

  * `-h`, `--help`: show this help message and exit
  * `-v`, `--verbose`: enable verbose output
  * `-q`, `--quiet`: disable non-error output
  * `-s`, `--silent`: disable all output
  * `-d`, `--debug`: enable exception traceback
  * `-w`, `--warning`: disable warning output
  * `-i INPUT`, `--input INPUT`: input file pathname
  * `-o OUTPUT`, `--output OUTPUT`: output file pathname
  * `-f FORMAT`, `--format FORMAT`: output data format
