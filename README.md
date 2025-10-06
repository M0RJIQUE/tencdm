## TEncDM: Understanding the Properties of Diffusion Model in the Space of Language Model Encodings

Abstract: *This paper presents the Text Encoding Diffusion Model
(TEncDM), a novel approach to diffusion modeling that operates in the space of pre-trained language model encodings.
In contrast to traditionally used embeddings, encodings integrate contextual information. In our approach, we also employ a transformer-based decoder, specifically designed to incorporate context in the token prediction process. We conduct a comprehensive examination of the influence of the
encoder, decoder, noise scheduler, and self-conditioning on
zero-shot generation. Furthermore, we compare TEncDM
with previous approaches on three conditional text generation tasks: QQP, XSum, and Wiki-Auto. The results show that
TEncDM exhibits superior performance compared to existing non-autoregressive diffusion models.*


## Requirements

* Python libraries: See [requirements.txt](./requirements.txt) for exact library dependencies. You can use the following commands with Miniconda3 to create and activate your Python environment:
  - `conda create --name tencdm python=3.9`
  - `conda activate tencdm`
  - `conda install pip`
  - `pip install -r requirements.txt`
  - `python -m spacy download en`

## Dataset loading

You can download any dataset used in the article with the following command:

```
python -m data.load --dataset_name=dataset_name
```

where `'dataset_name'` is one of the following:
 - `'wikipedia'`
 - `'rocstories'`
 - `'qqp'`
 - `'xsum'`
 - `'wiki-auto'`

By default your datasets will be saved in the folder `'datasets'` in your working directory. If you want to specify another path, use the argument `'dataset_path'`
## Statistics computation

For proper diffusion model training you need to compute dataset statistics for any particular dataset and encoder with the following command:

```
python -m data.make_statistics --dataset_name=dataset_name --encoder_name=encoder_name
```

where `'encoder_name'` is one the following:
 - `'bert-base-cased'`
 - `'t5-base'`
 - `'roberta-base'`
 - `'bart-base'`

Notice that default TEncDM encoder is `'bert-base-cased'`, but you can experiment with it


## Decoder training
Just as the statistics, you need to train a decoder for every pair of dataset and encoder. You can do this with the command:

```
python -m model.train_decoder --dataset_name=dataset_name --encoder_name=encoder_name
```

## Diffusion training

Finally, after training the decoder and computing dataset statistics, you can start training yout diffusion model. To train basic TEncdDM setup, run

```
torchrun --nproc_per_node=n train_diffusion.py --dataset_name=dataset_name --encoder_name=encoder_name
```

You will train the best TEncDM network with this script. But you can also test other hyperparameters, such as:

- `--emb` - wether to train using only embeddings, `default=False`
- `--scheduler` - noise scheduler to use,`default='sd'`. Can be one of the following: `'sqrt'`, `'cosine'`, `'sd'`
- `--coef_d` - coefficient for tangent scheduler, `default=9`
- `--swap_cfg_coef` - classifier free guidance coefficient, `default=0.0`
- `--project_name` - wandb project name, `default='tencdm'`


Notice: if you want to train your model on embeddings, you need to train decoder again with `--emb=True`
