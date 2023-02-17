"""
# https://nlpaug.readthedocs.io/en/latest/augmenter/word/spelling.html

Create classed for Data Augmentation. The goal is to use it for text data augmentation. 

These class allow to augmentate data with different ways: with spelling, language generative models, masked language models, back translation augmenter, word embeddings, abstractive summarization and synonyms.  

Using object-oriented programming with an abstract class called Data Augmentation

* DataAugmentationAbstract:
* __init__(method, stopwords): method is a string, stopwords is a list of strings
* augment(texts, num_samples, num_workers): text is an string, a list of strings or a pandas Series and n_samples is an integer, and num_workers is an integer.
* _text_augment_(text, num_samples): text is a string and num_samples is an integer.
* _list_augment_(texts, num_samples): texts is a list of strings and num_samples is an integer.
* _pandas_augment_(texts, num_samples): texts is a pandas Series and num_samples is an integer. 

Then create the following classes that inherit from DataAugmentationAbstract:

* DataAugmentationSpelling
** __init__(method, stopwords, aug_percent, tokenizer): method is a string ("keyboard", "ocr", "random", "misspelling"),  min_words is an integer or a float and max_words is an integer or a float.
* DataAugmentationGenerative
** __init__(method, stopwords, max_words, model, device, temperature, top_k, top_p, device, frequency_penalty, presence_penalty)
* DataAugmentationMasked
** __init__(method, stopwords, max_words, model, device, top_k)
* DataAugmentationBackTranslation
** __init__(method, stopwords, max_words, model, device, temperature, top_k)
* DataAugmentationWordEmbeddings
** __init__(method, stopwords, aug_percent, model, device, top_k)
* DataAugmentationAbstractiveSummarization
** __init__(method, stopwords, max_words, model, device, temperature, top_k)
* DataAugmentationSynonyms
** __init__(method, stopwords, aug_percent, dictionary, top_k)

Generate the code. 

"""
