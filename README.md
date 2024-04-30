# ViTLP
This repository releases code of paper [**Visually Guided Generative Text-Layout Pre-training for Document Intelligence** (NAACL-2024)](https://arxiv.org/abs/2403.16516).


We provide the pre-trained checkpoint **ViTLP-medium** (380M). The pre-trained ViTLP model can natively perform OCR text localization and recognition, which is accessible at [Huggingface](https://huggingface.co/veason/ViTLP-medium/tree/main). Clone (or download) ViTLP checkpoint weight to the directory `./ckpts/ViTLP-medium`.

<pre><code>pip install -r requirements.txt
git clone ViTLP && cd ViTLP

# Clone ViTLP-medium checkpoint
mkdir ckpts
git clone https://huggingface.co/veason/ViTLP-medium ckpts</code></pre>


# Demo
With the checkpoint and dependencies set (see requirements.txt), run the demo as

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
