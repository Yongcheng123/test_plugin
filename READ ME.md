# Requirements

  * ChimeraX

> Download [ChimeraX](https://www.cgl.ucsf.edu/chimerax/download.html). 

  * Dependencies
>   ChimeraX-Core version="~=1.0"

>   Python3

>   [Torch >=1.4.0](https://pytorch.org/)

>   [numpy >=1.16.4](https://pypi.org/project/numpy/)

>   [importlib-metadata >=0.17](https://pypi.org/project/importlib-metadata/#history)

# Install the bundle

# Sample Input & Output

If run without command line arguments, it will be use ./data/sensors-2018.12.26-no-labels.txt
using

```
python3 main.py
```

the following usage message will be displayed. 

```
   Time (sec)  core-1  core-2  core-3  core-4
0           0    61.0    63.0    50.0    58.0
1          30    80.0    81.0    68.0    77.0
2          60    62.0    63.0    52.0    60.0
3          90    83.0    82.0    70.0    79.0
4         120    68.0    69.0    58.0    65.0
```
Output files will be automatically generated to ./output/sensors-2018.12.26/***.txt

---

If run using argument, an input file path need to be added

```
python3 main.py --file ./data/(input file path)
```

The displayed message will *simliar* to

```
   Time (sec)  core-1  core-2  core-3  core-4
0           0    61.0    63.0    50.0    58.0
1          30    80.0    81.0    68.0    77.0
2          60    62.0    63.0    52.0    60.0
3          90    83.0    82.0    70.0    79.0
4         120    68.0    69.0    58.0    65.0

```

Output files will be automatically generated to ./output/(input_file_info)/***.txt

---

Sample output file
```
0     <= x < 30   ; y_0  = 61.0000 + 0.6333x; interpolation
30    <= x < 60   ; y_1  = 98.0000 + -0.6000x; interpolation
60    <= x < 90   ; y_2  = 20.0000 + 0.7000x; interpolation
90    <= x < 120  ; y_3  = 128.0000 + -0.5000x; interpolation
120   <= x < 150  ; y_4  = 12.0000 + 0.4667x; interpolation
...
...
...
35550 <= x < 35580; y_1185 = -20084.0000 + 0.5667x; interpolation
35580 <= x < 35610; y_1186 = -3480.0000 + 0.1000x; interpolation
0     <= x < 35610; y_least_square = -0.0001 + 77.1459x; least-squares
```