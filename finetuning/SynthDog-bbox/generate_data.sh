# pip install synthtiger
# pip install Pillow==9.5.0

synthtiger -o synthetic_images -c 10 -v template.py SynthDoG config.yaml

python preprocess.py # Convert data to the format of [[word, [x1, y1, x2, y2]]...] and perform visualization.
