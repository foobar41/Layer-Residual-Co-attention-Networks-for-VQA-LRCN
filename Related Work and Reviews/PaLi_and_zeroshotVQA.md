# PaLi Overview

PaLI handles both images and text together so that it can perform tasks like:

- describing images (captioning)
- answering questions about images
- understanding text within images

### Key Innovations

- Uses simple interface:
    - Input: Images and text
    - Output: Text
- Doesn’t build from scratch, it instead uses existing pre-trained models like mT5 for text processing and ViTs for image processing
- Scaling both vision and language components together is important for performance. Therefore to balance them (LLMs are MUCH larger than ViTs), they created a massive 4-billion parameter ViT to better balance them. (How do we find this ViT, and how much compute resource would this require? Are we training them?)
- Trained on ENOURMOUS multilingual dataset (10 billion images with texts in over 100 languages) (Can we reduce to just one language? DO we need to train on this dataset? DId they use this for training or finetuning?)

Task is frames as a general VQA task: “image+query to answer” framework is modelled

### Requirements

- mT5 pretrained model
    - Available on [huggingface](https://huggingface.co/docs/transformers/en/model_doc/mt5), released by google
    - They use the 13B-parameter model mT5-XXL
- Vision Model
    - They use the 2B-parameter ViT-G model and train a 4B-parameter model called ViT-e
    - Need to find the model
- To train this model (PaLI-17B) , they created a high-volume dataset called WebLI
    - 10 billion images and 10s of billions of image-text pairs
    - Dataset is not available for public use (Only for internal Google use)

They train all the 3 models (PaLI of different sizes, some using just pretrained models) for one epoch on the main dataset, so we can’t replicate any of the results given that we don’t have access to the dataset

# Zero Shot VQA with Frozen LLMs Overview

Introduces Img2LLM that enables LLMs to perform zero-shot visual question answering without finetuning or end-to-end training

This is done by converting image content into textual question-answer pairs (I think this will mostly be LLMs instead of CV)

## How they built it

- They used an off-the-shelf question-relevant caption generation module to generate captions
- Nouns, verbs, adjectives etc were extracted from these captions using spacy as potential answers
- then use any question generation network to generate questions for each answer
    - This model is entirely textual based

## **How to Implement from Scratch**

1. **Set up the vision components**:
    - Implement BLIP for caption generation and image-question matching
    - Implement GradCAM for generating relevance maps from cross-attention layers
2. **Implement answer extraction**:
    - Generate multiple captions for the image
    - Use a parser (like spaCy) to extract noun phrases, verb phrases, etc.
    - Count frequency of extracted answers across captions
3. **Implement question generation**:
    - Either use templates based on answer POS
    - Or finetune T5-large on textual QA datasets (SQuAD2.0, MultiRC, BookQA, etc.)
4. **Implement caption selection**:
    - Generate GradCAM heatmaps for image regions relevant to the question
    - Sample image patches based on relevance scores
    - Generate diverse captions from these patches
    - Filter captions based on similarity scores
5. **Construct the prompt**:
    - Format instruction, captions, and QA pairs
    - Add the actual question at the end
    - Send to any off-the-shelf LLM (OPT, GPT-J, etc.)