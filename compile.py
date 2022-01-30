import os

os.system(
"cythonize main.pyx -3 --inplace &&\
rm -rf build main.cpp"
)

import main
