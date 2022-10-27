from spanish_nlp import preprocess
sp = preprocess.SpanishPreprocess(
    lower=True,
    remove_url=True,
    remove_hashtags=True,
    preserve_emoticons=True,
    preserve_emojis=True,
    convert_emoticons=False,
    convert_emojis=False,
    normalize_inclusive_language=True,
    reduce_spam=True,
    remove_vowels_accents=True,
    remove_punctuation=True,
    remove_unprintable=True,
    remove_numbers=True,
    remove_stopwords=False,
    stopwords_list=None,
    stem=False,
)

test_text = """𝓣𝓮𝔁𝓽𝓸 𝓭𝓮 𝓹𝓻𝓾𝓮𝓫𝓪

Este es un texto de prueba :) a continuación les mostraré un poema llamado "Los perros románticos" 🤭👀😅

https://www.poesi.as/rb9301.htm

Me gusta la LINGÜÍSTICA y los ñandúes… También los pingüinos 🐧🐧🐧. 

Tengo una deuda de $10.000.000.000, pero tengo 2000 vacas. Mi teléfono es +5698791045"""

print("Original text:")
print(test_text)
print("-"*5)
print("Preprocessed text:")
print(sp.transform(test_text))
