from spanish_nlp import preprocess
sp = preprocess.SpanishPreprocess(
        lower=True,
        remove_url=True,
        remove_hashtags=False,
        split_hashtags=True,
        remove_emoticons=False,
        remove_emojis=False,
        convert_emoticons=True,
        convert_emojis=True,
        normalize_inclusive_language=False,
        reduce_spam=True,
        remove_vowels_accents=True,
        remove_punctuation=True,
        remove_unprintable=True,
        remove_numbers=True,
        remove_stopwords=True,
        stopwords_list="nltk",
        lemmatize=False,
        stem=False,
)

test_text = """𝓣𝓮𝔁𝓽𝓸 𝓭𝓮 𝓹𝓻𝓾𝓮𝓫𝓪

Este es un texto de prueba :) a continuación les mostraré un poema de Roberto Bolaño llamado "Los perros románticos" 🤭👀😅

https://www.poesi.as/rb9301.htm

Me gusta la LINGÜÍSTICA y los ñandúes… También los pingüinos 🐧🐧🐧. #VivanLosPinguinos #SiSeñor #PinguinosDelMundoUníos #ÑanduesDelMundoTambién

Tengo una deuda de $10.000.000.000, pero tengo 2000 vacas. Mi teléfono es +5698791045"""

print("Original text:")
print(test_text)
print("==="*30)
print("Preprocessed text:")
print(sp.transform(test_text, debug=False))
