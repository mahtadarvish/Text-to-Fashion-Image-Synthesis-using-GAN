# Text-to-Fashion-Image-Synthesis-using-GAN

Pytorch Implementation of our paper [Towards the Efficiency of the Fusion Step in Language-Based Fashion Image Editing] (https://ieeexplore.ieee.org/document/9780492)

Download a pretrained [English](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.zip) word vectors. Unzip it and move `wiki.en.bin` to `fasttext_models/`

## Datasets download
Fashion Synthesis: download `language_original.mat`, `ind.mat` and `G2.zip` from [here](https://drive.google.com/drive/folders/0B7EVK8r0v71pTHhMenkxbE9fTVk)

1. preprocess training data by runing `python2 process_fashion_data.py

2.train visual-semantic embedding model 
python2 train_text_embedding.py \
    --img_root ./datasets \
    --caption_root ./datasets/FashionGAN_txt \
    --trainclasses_file trainclasses.txt \
    --fasttext_model ./fasttext_models/wiki.en.bin \
    --save_filename ./models/text_embedding_fashion.pth
    
3. train
python2 train.py \
    --img_root ./datasets \
    --caption_root ./datasets/FashionGAN_txt \
    --trainclasses_file trainclasses.txt \
    --fasttext_model ./fasttext_models/wiki.en.bin \
    --text_embedding_model ./models/text_embedding_fashion.pth \
    --save_filename ./models/fashion_res_lowrank_64.pth \
    --use_vgg \
    --fusing_method lowrank_BP
    
### Other fusing methods

You can modify `--fusing_method` to train the model by different fusing methods: `lowrank_BP`, `FiLM` and default is `concat`

## Test
python2 test.py \
    --img_root ./test/fashion \
    --text_file ./test/text_fashion.txt \
    --fasttext_model ./fasttext_models/wiki.en.bin \
    --text_embedding_model ./models/text_embedding_fashion.pth \
    --generator_model ./models/fashion_res_lowrank_64.pth \
    --output_root ./test/result_fashion \
    --use_vgg \
    --fusing_method lowrank_BP
