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

https://user-images.githubusercontent.com/67181111/141269015-6be6a34d-1930-4dcb-ab13-555c601d2b04.mp4

https://user-images.githubusercontent.com/67181111/141268813-df641fa1-4fa6-47bf-987e-463bdfc4e828.mp4
