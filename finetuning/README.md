# Finetuning ViTLP
To begin with, we get pre-trained ViTLP checkpoint and OCR data ready. For instance, we clone the pre-trained ViTLP checkpoint at `../ckpts/ViTLP-medium` and put the OCR data at [SynthDog-bbox/preprocessed_data](https://github.com/Veason-silverbullet/ViTLP/tree/main/finetuning/SynthDog-bbox/preprocessed_data/data). Of course, you can place the ViTLP checkpoint and OCR data files anywhere.



## Preprocess Dataset for Finetuning
Run `preprocess_data.py` to preprocess the finetuning dataset:

<pre><code>process_text_bbox_data(
    metadata_dir = 'SynthDog-bbox/preprocessed_data/data',
    image_dir = 'SynthDog-bbox/preprocessed_data/images',
    preprocessed_data_dir = 'text_bbox_data'
)</code></pre>

- `metadata_dir`: The OCR metadata is prepared in the required JSON format. Please refer to data-format examples at [./SynthDog-bbox/preprocessed_data/data](https://github.com/Veason-silverbullet/ViTLP/tree/main/finetuning/SynthDog-bbox/preprocessed_data/data).

- `image_dir`: The finetuning document images. Please refer to image examples at [./SynthDog-bbox/preprocessed_data/images](https://github.com/Veason-silverbullet/ViTLP/tree/main/finetuning/SynthDog-bbox/preprocessed_data/images).

- `preprocessed_data_dir`: The preprocessed finetuning dataset location.



## Finetuning ViTLP
Run `finetune.py` to finetune ViTLP:
<pre><code># For example, finetune ViTLP with 4 GPUs and Deepspeed Zero-1, saving the checkpoint at `./outputs`
deepspeed --num_nodes 1 --num_gpus 4 finetune.py --deepspeed_config=zero1_fp16.json --output_dir=outputs</code></pre>
