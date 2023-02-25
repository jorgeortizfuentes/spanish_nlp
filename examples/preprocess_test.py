from spanish_nlp import preprocess

sp = preprocess.SpanishPreprocess(
    lower=True,
    remove_url=True,
    remove_hashtags=False,
    split_hashtags=True,
    normalize_breaklines=True,
    remove_emoticons=False,
    remove_emojis=False,
    convert_emoticons=False,
    convert_emojis=False,
    normalize_inclusive_language=False,
    reduce_spam=True,
    remove_reduplications=True,
    remove_vowels_accents=True,
    remove_multiple_spaces=True,
    remove_punctuation=True,
    remove_unprintable=True,
    remove_numbers=True,
    remove_stopwords=False,
    stopwords_list=None,
    lemmatize=False,
    stem=False,
    remove_html_tags=True,
)

test_text = """𝓣𝓮𝔁𝓽𝓸 𝓭𝓮 𝓹𝓻𝓾𝓮𝓫𝓪

<b>Holaaaaaaaa </b>, este es un texto de prueba :) a continuación les mostraré un poema de Roberto Bolaño llamado "Los perros románticos" 🤭👀😅

https://www.poesi.as/rb9301.htm

Me gusta la LINGÜÍSTICAAAA, los pandas y los ñandúes… También los pingüinosssss 🐧🐧🐧 🐧. #VivanLosPinguinos #SíSeñor #PinguinosDelMundoUníos #ÑanduesDelMundoTambién

Si colaboras con este código te puedes ganar $10.000.000.000. O tal vez 2000 vacas. Mi teléfono es +569123456789"""

print(sp.transform(test_text, debug=True))
