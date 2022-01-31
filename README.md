# Ascii-Render
A 3d rasterizer that runs on the terminal.


## Installation / Usage
- clone the repo: `$ git clone https://github.com/zyugyzarc/Ascii-Render.git`
- install cython `$ pip3 install cython`
- compile and run `main.pyx`:
  - `$ cythonize main.pyx -3 --inplace && rm -rf build main.c`
  or run `python3 compile.py [obj file]`
* the framerate can be set using the `FPS` env variable

## Example footage

![](https://raw.githubusercontent.com/zyugyzarc/Ascii-Render/main/.github/color_example.gif)

![](https://raw.githubusercontent.com/zyugyzarc/Ascii-Render/main/.github/suzane_mouse.gif)

![](https://raw.githubusercontent.com/zyugyzarc/Ascii-Render/main/.github/mario.gif)
