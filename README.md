# ViTLP
This repository releases code of paper [**Visually Guided Generative Text-Layout Pre-training for Document Intelligence** (NAACL-2024)](https://arxiv.org/abs/2403.16516).


We provide the pre-trained checkpoint ViTLP-medium (380M) that can natively perform text localization and recognition, i.e., OCR. The ViTLP checkpoint weight is released at [Google Drive](https://drive.google.com/drive/folders/1OapAwZjnqoep9TzgjtyjYHgdeQzgCiKp?usp=sharing). Download and place the checkpoint weight at `./ckpts/ViTLP-medium`.


# Demo
With the checkpoint downloaded and dependencies installed (see requirements.txt), run the demo as

<pre><code>python ocr.py</code></pre>

Upload a document image and have a shot

![](misc/ocr-demo-1.png)

![](misc/ocr-demo-2.png)

See detailed inference code at `decode.py` and run batch decode by

<pre><code>bash decode.sh</code></pre>


# Preset FAQ
- Why is ViTLP-medium (380M)?

  When I commenced this project, it was on the eve of LLMs (precisely speaking, ChatGPT). ViTLP-base presented in our paper, is actually a rather small pre-trained model. We know it is expected to scale up ViTLP in this LLM era. However, the pre-training scale is commonly constrained by computation resources and the pre-training dataset scale, in which context ViTLP-medium (380M) is the largest pre-training scale so far we can support.

  Besides, this scale of ViTLP also brings inference sweetness including speed and memory usage. Typically, OCR on a page of a document image can be processed within 5~10 seconds in an Nvidia 4090, which is comparable to (and faster than) most OCR engines (and LLMs).


# Note
ViTLP is pronounced /ˈvai·tlp/ (vital). The first version of our paper was submitted to [OpenReview](https://openreview.net/forum?id=ARtBIBAmNR) in June 2023.
