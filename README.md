# nlp-text-subjectivity-conversion (WIP)

A machine learning project to convert text from having an objective voice to subjective, and vice versa. A WIP, please check back later!

Usage notes: The first time you run objectivity or subjectivity transformation on your system, the process will take a long time. This is because it is downloading and caching about 4GB worth of files. (said files will be in the `cached-GloVe` and `checkpoint` folders, so don't touch those!)

This is completely normal. Check console outputs to see the download progress.

requirements:

```
numpy
pytorch with CUDA support
gdown
tkinter
pickle
```

if you get `AssertionError: Torch not compiled with CUDA enabled`, try this:
```
pip uninstall torch
pip cache purge
pip install torch -f https://download.pytorch.org/whl/torch_stable.html
```