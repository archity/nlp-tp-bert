# TP BERT

Give it before the 31st of January 2022, 8am (Grenoble hour).

[here](https://drive.google.com/drive/folders/175XKNqc0bl5p1mp_17HozbLHIYTMRTjz?usp=sharing)

The exercises of this TP, the instructions and the code are entirely contained in the Notebook `tp_bert.ipynb`, which you will find in this directory. There are two methods to open and run it, the first of which is highly recommended because it saves you from installing software on your local machine. In addition, the execution of the code will be faster.

## Method 1: Google Colab

- Sign up for Google Colab (if you don't already have a Google account) .
- Once you have obtained access to a Colab file, you can download a Notebook using `file->Upload Notebook`. Download the file `tp_bert.ipynb`.
- To enable the use of GPUs for your notebook: `Runtime->Change runtime type->Hardware Accelerator->GPU` .

## Method 2: conda environment

Install [Miniconda3](https://docs.conda.io/en/latest/miniconda.html) (Anaconda3 should also work), then, go to this TP directory (avec `cd home/username/mypath`) and execute the following commands:

**Note**: `pip install transformers[torch]` instals the CPU version of Pytorch. If you have a GPU on your computer, first install PyTorch for GPU following instructions [here](https://pytorch.org/get-started/locally/#start-locally).

````
conda create --name bertenv python
conda activate bertenv
pip install transformers[torch]
conda install -c conda-forge notebook
conda install scikit-learn
conda install -c conda-forge ipywidgets
pip install tensorboardX
````

Now, you can open  `tp_bert.ipynb`, use `jupyter notebook` in work directory
