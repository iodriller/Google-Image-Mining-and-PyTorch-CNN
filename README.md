# Google Image Mining and PyTorch CNN

This is a fun project that allows automatically downloading "scraping" google images and training a Convolutional Neural Network.  

### Requirements

Install the requirements by running the code below at the command window, at the project's directory:

	pip install -r requirements.txt

For google image mining, install:

[chromedriver](http://chromedriver.storage.googleapis.com/index.html?path=2.9/)

At the example jupyter notebooks, replace <PATH> with the directory you put the "chromedriver.exe":

	DRIVER_PATH = r"<PATH>"

i.e.:

	DRIVER_PATH = r"C:\your_directory\chromedriver.exe"

### Examples

There are two jupyter notebooks that shows how to do image mining and running CNN:

- __[example_1:](https://github.com/iodriller/Google-Image-Mining-and-PyTorch-CNN/blob/master/example_1.ipynb)__ - Mines google images based on a list of keywords, then runs PyTorch CNN and returns the network. Also, provides an accuracy metric.
- __[example_2:](https://github.com/iodriller/Google-Image-Mining-and-PyTorch-CNN/blob/master/example_2.ipynb)__ - This is for google image search only. With just couple of lines, you can mine quite a bit images automatically.

### References

Inspired by these two sources:
1. [https://pythonprogramming.net/convolutional-neural-networks-deep-learning-neural-network-pytorch/](https://pythonprogramming.net/convolutional-neural-networks-deep-learning-neural-network-pytorch/)
2. [https://towardsdatascience.com/image-scraping-with-python-a96feda8af2d](https://towardsdatascience.com/image-scraping-with-python-a96feda8af2d)
