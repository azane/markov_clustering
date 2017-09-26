# Markov Clustering

## Implementation of the MCL algorithm in python

The MCL algorithm was developed by Stijn van Dongen at the University of Utrecht.

Details of the algorithm can be found on the [MCL homepage](https://micans.org/mcl/).

### Isn't there already a python module for this?

Yes and no.

There is a module available on pypy called python_mcl which implements this algorithm.
However, reading through the code reveals several issues:
  - The expansion and inflation operations are not performed in the correct order.
  - There is no support for sparse matrices
  - There is no pruning performed