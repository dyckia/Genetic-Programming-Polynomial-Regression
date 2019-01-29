**HOW TO RUN THIS PROGRAM**

1. open terminal
2. switch to python3 environment
3. go to the directory where this gp.py locates
4. make sure the *"A4_trainingSamples.txt"* file is in the same directory as this gp.py is
5. type in terminal:

> python3 gp.py pop_size tour_size gen

**WHERE:**

- pop_size as the population size
- muta_rate as the tournament selection size
- gen as the evolution epochs for the program to run


An example of command to run this program would be:

------------------------------------------------------------
> python3 gp.py 100 6 10000
------------------------------------------------------------

**REQUIREMENTS**
- Python 3.5 or higher environment with package installed
- Dependent Packages: sys, random, math, copy


**OTHERS**
- Others parameter that can be tuned within the source code:
    - maximum depth for tree growth depth & mutation (*max_depth=5*)
    - corssover rate (*c_rate=1*)
    - mutation rate (*m_rate=0.2*)
