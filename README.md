# nlp-text-subjectivity-conversion (WIP)

A machine learning project that greatly concerns text objectivity. It aims to, firstly, detect text objectivity, and then secondly, subjectify text and objectify them.

Google Colab link: https://colab.research.google.com/drive/1A9peY0Neqh93G6lFx6q8hleSNww-4bL9?usp=sharing

## Objectivity Detection
Our model is trained using Supervised methods to detect text objectvity. Refer to the above Colab notebook for our code. We attempted to infer objectivity through author sentiment, but the direct and conventional approach performs better (~75% vs ~90%). Dataset used can be found [here](http://www.cs.cornell.edu/people/pabo/movie-review-data/).

## Objectification and Subjectification
An RNN is used to objectify and subjectify text, and is compared with a naïve method of objectification and subjectification. This part is a WIP, please check back later!

## Dependencies:

```
numpy
pytorch
gdown
tkinter
pickle
sklearn
```
## Running the app
Run the following command inside the src directory:
```
python3 gui.py
```
Usage notes: The first time you run objectivity or subjectivity transformation on your system, the process will take a long time. This is because it is downloading and caching about 4GB worth of files. (said files will be in the `cached-GloVe` and `checkpoint` folders, so don't touch those!)

This is completely normal. Check console outputs to see the download progress.
