# Review on [SimVLM paper](https://arxiv.org/abs/2108.10904)
This paper proposes a simpler model for Vison Language Modelling with image and text as input and text as output. The model takes a encoder-decoder architecture like Neural Machine Translation model and inputs image text patches before text embeddings.

## Pretraining Task: PrefixLM
In this task, they basically divide the text into two parts of random length and send first part and patched image into the model. And the output is basically the second part of the text. The pretraining loss is just autoregresisve loss of generating the second sequence.

## Dataset
The authors use training set of a model called [ALIGN](https://research.google/blog/align-scaling-up-visual-and-vision-language-representation-learning-with-noisy-text-supervision/) which is not public but it is a very very big dataset(1.8B) crawled from internet where text is basically the alt text of the image. They also use another dataset called [Colossal Clean Crawled Campus](https://huggingface.co/datasets/allenai/c4) which has 2.15M data points. 

Since they didn't release the original model, there is another company called Kakao Brain which released a similar dataset called [COYO](https://github.com/kakaobrain/coyo-dataset#dataset-preview) which is a 700M datapoints and a different model trained on this got similar performance. See perf graphs [here](https://huggingface.co/kakaobrain/align-base#align-base-model).

## Can we use this for our project?

The model has a simplier architecture compared to modern VLMs. So maybe this model can be quick and fast although there is no details on inference time on the paper. But the model is not released so if we really wanna use we have to pretain our own model which will take days or even weeks in my opinion.

### TLDR
**No we can't since the model checkpoint is not released online. 
However there is a similar model that is mentioned above ALIGN which has a similar architecture to OpenAIs CLIP. But we cannot use SimVLM.**
