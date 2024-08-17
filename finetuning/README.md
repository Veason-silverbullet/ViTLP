# Fine-tuning ViTLP
To begin with, we get pre-trained ViTLP checkpoint and OCR data ready. For instance, we clone the pre-trained ViTLP checkpoint at `../ckpts/ViTLP-medium` and put the OCR data at [./SynthDog-bbox/preprocessed_data](https://github.com/Veason-silverbullet/ViTLP/tree/main/finetuning/SynthDog-bbox/preprocessed_data/data) and VQA data at [./DocVQA](https://github.com/Veason-silverbullet/ViTLP/tree/main/finetuning/DocVQA). Of course, you can place the ViTLP checkpoint and dataset files anywhere.



## Preprocess OCR Datasets for Fine-tuning
Run `preprocess_data.py` to preprocess the OCR fine-tuning dataset:

<pre><code>process_text_bbox_data(
    metadata_dir = './SynthDog-bbox/preprocessed_data/data',
    image_dir = './SynthDog-bbox/preprocessed_data/images',
    preprocessed_data_dir = './text_bbox_data'
)</code></pre>

- `metadata_dir`: The OCR metadata is prepared in the required JSON format. Please refer to data-format examples at [./SynthDog-bbox/preprocessed_data/data](https://github.com/Veason-silverbullet/ViTLP/tree/main/finetuning/SynthDog-bbox/preprocessed_data/data).

- `image_dir`: The finetuning document images. Please refer to image examples at [./SynthDog-bbox/preprocessed_data/images](https://github.com/Veason-silverbullet/ViTLP/tree/main/finetuning/SynthDog-bbox/preprocessed_data/images).

- `preprocessed_data_dir`: The preprocessed fine-tuning dataset location.


## Preprocess DocVQA Datasets for Fine-tuning
Initially, download and unzip the document images into `./DocVQA/documents` from [DocVQA Challenge](https://rrc.cvc.uab.es/?ch=17&com=downloads). Run `preprocess_docvqa_data.py` to preprocess the DocVQA fine-tuning dataset:

<pre><code>process_docvqa_train_data(data_dir = './DocVQA')</code></pre>

- `data_dir`: The DocVQA metadata is readily available at [./DocVQA/train-metadata.json](https://github.com/Veason-silverbullet/ViTLP/tree/main/finetuning/DocVQA/train-metadata.json).


## Fine-tuning ViTLP
- Run `./finetune.py` to fine-tune ViTLP on the OCR dataset:
<pre><code># Fine-tune ViTLP with 4 GPUs and Deepspeed Zero-1, saving the checkpoint at `./outputs`
deepspeed --num_nodes 1 --num_gpus 4 finetune.py --deepspeed_config=misc/zero1_fp16.json --output_dir=outputs</code></pre>


- Run `./finetune_docvqa.py` to fine-tune ViTLP on the DocVQA dataset:
<pre><code># Fine-tune ViTLP with gradient accumulation steps of 4, saving the checkpoint at `./DocVQA-outputs`
deepspeed --num_nodes 1 --num_gpus 4 finetune_docvqa.py --batch_size=8 --deepspeed_config=misc/zero1_fp16-grad_acc-4.json --output_dir=DocVQA-outputs</code></pre>
