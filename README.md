# Ascii-Render
A 3d rasterizer that runs on the terminal.

## Installation / Usage
- clone the repo: `$ git clone https://github.com/zyugyzarc/Ascii-Render.git`
- install cython `$ pip3 install cython`
- compile and run `main.pyx`:
  - `$ cythonize main.pyx -3 --inplace && rm -rf build main.c`
  - `$ python3 -c "import main" [path/to/obj/file]`
  or run `python3 compile.py [path/to/obj/file]`
* the framerate can be set using the `FPS` env variable
* color can be disabled by setting the `NOCOLOR` env variable to any value *[Note: this increases speed]*

Note: if your model contains textures, the texture size must be set manually in `main.pyx` at line 31.

## Example footage

![](https://raw.githubusercontent.com/zyugyzarc/Ascii-Render/main/.github/textured.gif)

![](https://raw.githubusercontent.com/zyugyzarc/Ascii-Render/main/.github/color_example.gif)

![](https://raw.githubusercontent.com/zyugyzarc/Ascii-Render/main/.github/suzane_mouse.gif)

![](https://raw.githubusercontent.com/zyugyzarc/Ascii-Render/main/.github/minecraft.gif)

![](https://raw.githubusercontent.com/zyugyzarc/Ascii-Render/main/.github/mario.gif)

