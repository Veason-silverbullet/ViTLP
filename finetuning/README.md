# Fine-tuning ViTLP
To begin with, we get pre-trained ViTLP checkpoint and OCR data ready. Clone the pre-trained ViTLP checkpoint at `../ckpts/ViTLP-medium` and put the OCR data at [./SynthDog-bbox/preprocessed_data](https://github.com/Veason-silverbullet/ViTLP/tree/main/finetuning/SynthDog-bbox/preprocessed_data/data) and VQA data at [./DocVQA](https://github.com/Veason-silverbullet/ViTLP/tree/main/finetuning/DocVQA). Of course, you can place the ViTLP checkpoint and dataset files anywhere.



## Preprocess OCR Datasets for Fine-tuning
Run `preprocess_data.py` to preprocess the OCR fine-tuning dataset:

<pre><code>process_text_bbox_data(
    metadata_dir = './SynthDog-bbox/preprocessed_data/data',
    image_dir = './SynthDog-bbox/preprocessed_data/images',
    preprocessed_data_dir = './text_bbox_data'
)</code></pre>

- `metadata_dir`: The OCR metadata is prepared in the required JSON format. Please refer to data-format examples at [./SynthDog-bbox/preprocessed_data/data](https://github.com/Veason-silverbullet/ViTLP/tree/main/finetuning/SynthDog-bbox/preprocessed_data/data).

- `image_dir`: The fine-tuning document images. Please refer to image examples at [./SynthDog-bbox/preprocessed_data/images](https://github.com/Veason-silverbullet/ViTLP/tree/main/finetuning/SynthDog-bbox/preprocessed_data/images).

- `preprocessed_data_dir`: The preprocessed fine-tuning dataset location.



## Preprocess DocVQA Datasets for Fine-tuning
1. Initially, download and unzip the document images into `./DocVQA/documents` from [DocVQA Challenge](https://rrc.cvc.uab.es/?ch=17&com=downloads).

2. (Optional) Download the QA answer pairs into `./DocVQA/train_v1.0_withQT.json` from [DocVQA Challenge](https://rrc.cvc.uab.es/?ch=17&com=downloads). We provided multi-source DocVQA OCR results on [Google Drive](https://drive.google.com/drive/folders/1AqVlqT0EP17wxxQjO_wcQwn2U7hIK1bh?usp=sharing). Then run [./DocVQA/link.py](https://github.com/Veason-silverbullet/ViTLP/tree/main/finetuning/DocVQA/link.py) with the multi-source OCR results to derive the metadata of answer bounding-boxes at [./DocVQA/train-metadata.json](https://github.com/Veason-silverbullet/ViTLP/tree/main/finetuning/DocVQA/train-metadata.json) which we have got it prepared.

3. Run `preprocess_docvqa_data.py` to preprocess the DocVQA fine-tuning dataset:

   <pre><code>process_docvqa_train_data(data_dir = './DocVQA')</code></pre>

- `data_dir`: The DocVQA metadata is readily available at [./DocVQA/train-metadata.json](https://github.com/Veason-silverbullet/ViTLP/tree/main/finetuning/DocVQA/train-metadata.json).



## Fine-tuning ViTLP
1. Run `./finetune.py` to fine-tune ViTLP on the OCR dataset:
   <pre><code># Fine-tune ViTLP with 4 GPUs and Deepspeed Zero-1, saving the checkpoint at `./outputs`
   deepspeed --num_nodes 1 --num_gpus 4 finetune.py --deepspeed_config=misc/zero1_fp16.json --output_dir=outputs</code></pre>


2. Run `./finetune_docvqa.py` to fine-tune ViTLP on the DocVQA dataset:
   <pre><code># Fine-tune ViTLP with gradient accumulation steps of 16, saving the checkpoint at `./DocVQA-outputs`
   deepspeed --num_nodes 1 --num_gpus 4 finetune_docvqa.py --batch_size=8 --deepspeed_config=misc/zero1_fp16-grad_acc-16.json --output_dir=DocVQA-outputs</code></pre>


## VQA Inference
Run `./inference_docvqa.py` to perform VQA with a fine-tuned ViTLP VQA model:
<pre><code># Given the fine-tuned ViTLP checkpoint at `--vqa_finetuned_model=./ViTLP-DocVQA` and a VQA image at `--image`, run inference code by
python inference_docvqa.py \
       --vqa_finetuned_model=./ViTLP-DocVQA \
       --image=./DocVQA/nkbl0226_1.png \
       --question="What is name of university?"</code></pre>
