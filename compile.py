import os

os.system(
"cythonize main.pyx -3 --inplace &&\
rm -rf build main.c* main.pyx"
)

import main
