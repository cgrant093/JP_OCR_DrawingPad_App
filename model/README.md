
The files in this folder are to training and storing the PyTorch models used to guess the Japanese character being drawn on the tablet. 

My process takes heavy inspiration from the one found in the paper ***Deep Learning for Classical Japanese Literature*** by Clanuwat et al. ([2018](https://arxiv.org/abs/1812.01718)). 

However, I will be implementing some differences in the process:

1. Utilize PyTorch in place of TensorFlow. 
2. Create a model that can recognize both Kana and (not-rare) Kanji
3. Only create the model trained for modern Kanji (do not need to create the Kuzushiji OCR)

To do this, I will follow the generic steps of:

1. Use a VAE (Variational Autoencoder) to create more training data from the kanakanji_assets PNG files.
2. Create a Sketch-RNN (or something similar) to recognize the characters.

<br/>

# Current Progress

## A. Create more training data using a VAE (Variational Autoencoder)
This is the first time I am implementing a VAE. As I was reading through the paper and other internet resources on VAEs, I came across a informative post by Hunter Heidenreich titled [Modern PyTorch Techniques for VAEs: A Comprehensive Tutorial](https://hunterheidenreich.com/posts/modern-variational-autoencoder-in-pytorch/). I am very grateful for this resource as I feel it has done a great job at introducing me to the concept and giving near-complete code examples on how to implement. It has introduced me to PyTorch's SummaryWriter, which I was not aware of, it seems incredibly useful for logging all sorts of information about the PyTorch model and its training progress. 

Some of his code blocks had a couple typos and/or reference to some of his custom objects/functions that were not presented in this article, and I needed to extrapolate/interpolate those on my own. I also plan on changing the encoder and decoder sequences to have Conv2d layers rather than the Linear layers he's utilizing. I am not sure why Linear layers were being used for images, as they lose the spatial awareness that makes Conv2d layers incredibly useful at recognizing 2D image patterns. Additionally, I need to figure out how to effectively turn my custom dataset into a PyTorch Dataset to properly utilize the DataLoader. 

<br/>

# Future Work

## A. Important model decision
I have been thinking that it *could* make sense to create one OCR per stroke count. Of all the characters I have basic training data for, the stroke counts range from 1 to 30. In my specific case, this *could* make sense because I will try to recognize characters in "real" time from a drawing tablet. This means I know exactly how many strokes have gone into each character. Making 30 different OCR models *could* speed up the training time and accuracy of the guess. However, I don't plan on dynamically changing the model's parameters depending on the stroke count, so I would affectively be increasing the number of "total model parameters" by a factor of 30, and there are a couple of other concerns/questions. Additionally, my limited knowledge of VAEs is very naive, if I implement 30 separate OCRs, do I also need to implement 30 separate VAEs to create the training data with?

Pros of 1 OCR per character stroke number:

- Potentially faster training times
- More accurate guesses as there's a much smaller pool of characters to look at
- Maybe faster at recognizing each character

Cons of 1 OCR per character stroke number:

- Increasing total OCR model parameters by a factor of 30
- If there is accidental doubled or hopping strokes, it will not recognize the character because it's in the different stroke count pool
- Maybe need to create 30 VAEs for the training data

I think I need to carefully think through my options.

## B. VAE
To-do:

1. Change the Linear layers to Conv2d layers
2. Change the aspects of the code (both upstream and downstream) that are affected by the above change
3. Turn my custom dataset of kana-kanji PNG files into a PyTorch Dataset subclass

Question(s):

1. Many VAEs implement a class/category label. Do I need on of these? If so, is my class label like stroke count, or the individual character? Or is it kana vs kanji? A couple of these do not seem like good "classes".

## C. OCR
Just the entire process. Following the Clanuwat et. al. (2018) paper, the OCR should be something like a Sketch-RNN. 

