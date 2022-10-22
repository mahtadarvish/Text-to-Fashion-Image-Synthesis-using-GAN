import os
#import argparse
from fasttext import FastText
from PIL import Image
import io
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.utils import save_image

from model import VisualSemanticEmbedding
from model import Generator
from data import split_sentence_into_words
import numpy as np

img_root = './test/fashion'
text_file = './test/text_fashion.txt'
fasttext_model = './fasttext_models/wiki.en.bin'
text_embedding_model = './models/text_embedding_fashion.pth'
embed_ndim = 300
generator_model = './models/fashion_res_lowrank_64.pth'
use_vgg = True
output_root = './images'
no_cuda = False
fusing_method = 'lowrank_BP'



if not no_cuda and not torch.cuda.is_available():
    print('Warning: cuda is not available on this machine.')
    no_cuda = True


if __name__ == '__main__':
    print('Loading a pretrained fastText model...')
    word_embedding = FastText.load_model(fasttext_model)

    print('Loading a pretrained model...')

    txt_encoder = VisualSemanticEmbedding(embed_ndim)
    txt_encoder.load_state_dict(torch.load(text_embedding_model))
    txt_encoder = txt_encoder.txt_encoder

    G = Generator(use_vgg=use_vgg, fusing=fusing_method)
    G.load_state_dict(torch.load(generator_model))
    

    

    G.train(False)

    if not no_cuda:
        txt_encoder 
        G 

    transform = transforms.Compose([
        transforms.Scale(70),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    transform2 = transforms.Compose([
        transforms.Scale(70),
        transforms.CenterCrop(64),
        transforms.ToTensor()
    ])

    print('Loading test data...')
    filenames = sorted(os.listdir(img_root))
    img = []
    img_save = []
    for fn in filenames:
        im = Image.open(os.path.join(img_root, fn))
        im_save = transform2(im)
        im = transform(im)
        img.append(im)
        img_save.append(im_save)
    img = torch.stack(img)
    img_save = torch.stack(img_save)
    save_image(img_save, os.path.join(output_root, 'original.jpg'),nrow=6, padding=0)
    img = Variable(img, volatile=True)

    html = '<html><body><h1>Manipulated Images</h1><table border="1px solid gray" style="width=100%"><tr><td><b>Description</b></td><td><b>Image</b></td></tr>'
    html += '\n<tr><td>ORIGINAL</td><td><img src="{}"></td></tr>'.format('original.jpg')
    with open(text_file, 'r') as f:
        texts = f.readlines()

    for i, txt in enumerate(texts):
        txt = txt.replace('\n', '')
        desc = split_sentence_into_words(txt)
        desc = torch.Tensor([word_embedding.get_word_vector(w).astype(np.float64) for w in desc])
        desc = desc.unsqueeze(1)
        desc = desc.repeat(1, img.size(0), 1)
        desc = Variable(desc, volatile=True)

        _, txt_feat = txt_encoder(desc)
        txt_feat = txt_feat.squeeze(0)
        output, _ = G(img, txt_feat)

        out_filename = 'output_%d.jpg' % i
        save_image((output.data + 1) * 0.5, os.path.join(output_root, out_filename),nrow=6, padding=0)
        html += '\n<tr><td>{}</td><td><img src="{}"></td></tr>'.format(txt, out_filename)

    with open(os.path.join(output_root, 'index.html'), 'w') as f:
        f.write(html)
    print('Done. The results were saved in %s.' % output_root)