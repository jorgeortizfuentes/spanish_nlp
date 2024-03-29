import os

DETERMINANTES = [
    "el",
    "la",
    "los",
    "las",
    "un",
    "una",
    "unos",
    "unas",
    "esta",
    "estas",
    "este",
    "estos",
    "esa",
    "esas",
    "ese",
    "esos",
    "aquel",
    "aquella",
    "aquellas",
    "aquellos",
    "mi",
    "tu",
    "su",
    "mis",
    "tus",
    "sus",
    "nuestro",
    "nuestra",
    "nuestros",
    "nuestras",
    "vuestro",
    "vuestra",
    "vuestros",
    "vuestras",
    "su",
    "sus",
    "cuyo",
    "cuya",
    "cuyos",
    "cuyas",
    "cuánto",
    "cuántos",
    "cuánta",
    "cuántas",
    "qué",
    "tan",
    "menos",
    "más",
    "algún",
    "alguna",
    "algunos",
    "algunas",
    "ningún",
    "ninguna",
    "ningunos",
    "ningunas",
]

PREPOSICIONES = [
    "a",
    "ante",
    "bajo",
    "cabe",
    "con",
    "contra",
    "de",
    "desde",
    "durante",
    "en",
    "entre",
    "hacia",
    "hasta",
    "mediante",
    "para",
    "por",
    "según",
    "sin",
    "so",
    "sobre",
    "tras",
    "versus",
    "vía",
]

PRONOMBRES = [
    "yo",
    "tú",
    "vos",
    "vo",
    "él",
    "ella",
    "ello",
    "elle",
    "usted",
    "nosotros",
    "nosotras",
    "nosotres",
    "vosotros",
    "vosotras",
    "vosotres",
    "ellos",
    "ellas",
    "ustedes",
    "elles",
    "mí",
    "ti",
    "sí",
    "consigo",
]

CONJUNCIONES = [
    "y",
    "ni",
    "sino",
    "tanto",
    "como",
    "que",
    "pero",
    "mas",
    "empero",
    "mientras",
    "o",
    "bien",
    "ya",
]

default_stopwords = list(set(DETERMINANTES + PREPOSICIONES + PRONOMBRES + CONJUNCIONES))

extended_stopwords = [
    "cómo",
    "propia",
    "por qué",
    "será",
    "uso",
    "cinco",
    "hubiera",
    "usa",
    "emplean",
    "todavía",
    "mal",
    "dan",
    "quiénes",
    "sólo",
    "tú",
    "para",
    "hicieron",
    "podriais",
    "apenas",
    "aquello",
    "manera",
    "trabaja",
    "arriba",
    "fueran",
    "tus",
    "unos",
    "está",
    "haces",
    "muchas",
    "proximo",
    "ellas",
    "aún",
    "aquella",
    "no",
    "has",
    "hoy",
    "delante",
    "hubiese",
    "encuentra",
    "mediante",
    "seamos",
    "suyos",
    "tendríais",
    "través",
    "mucho",
    "mas",
    "toda",
    "ciertos",
    "habrías",
    "nada",
    "bien",
    "ello",
    "son",
    "su",
    "varios",
    "éramos",
    "detrás",
    "tendríamos",
    "eran",
    "nadie",
    "poco",
    "aun",
    "conseguimos",
    "pais",
    "mi",
    "estuviste",
    "intentar",
    "como",
    "decir",
    "ellos",
    "estuvo",
    "claro",
    "tambien",
    "porque",
    "hacer",
    "fuera",
    "última",
    "ningunos",
    "seré",
    "primera",
    "tuyo",
    "tendrán",
    "éstos",
    "estaréis",
    "aqui",
    "han",
    "tendrías",
    "cuando",
    "vuestras",
    "fueseis",
    "tenga",
    "una",
    "sabemos",
    "hago",
    "otro",
    "largo",
    "aquellas",
    "mis",
    "deprisa",
    "míos",
    "empleas",
    "hecho",
    "mayor",
    "allí",
    "estuvisteis",
    "estábamos",
    "ocho",
    "habidas",
    "hacemos",
    "queremos",
    "propio",
    "mientras",
    "estuvieseis",
    "u",
    "estaba",
    "quizá",
    "grandes",
    "qué",
    "usamos",
    "breve",
    "siempre",
    "muy",
    "ojalá",
    "nuestras",
    "actualmente",
    "hacen",
    "aquéllas",
    "días",
    "hubierais",
    "últimas",
    "estaría",
    "seréis",
    "dicho",
    "habíamos",
    "gueno",
    "momento",
    "es",
    "bajo",
    "tendría",
    "estaremos",
    "lejos",
    "estemos",
    "ojala",
    "nuevas",
    "lugar",
    "todavia",
    "nunca",
    "estaré",
    "estabas",
    "ése",
    "tuya",
    "he",
    "mios",
    "indicó",
    "tengo",
    "pues",
    "hace",
    "podrá",
    "me",
    "nuestra",
    "hacerlo",
    "despues",
    "quizás",
    "estadas",
    "ésta",
    "estuviesen",
    "intentais",
    "los",
    "tendréis",
    "tu",
    "voy",
    "además",
    "sola",
    "tengas",
    "tienes",
    "estáis",
    "general",
    "último",
    "estan",
    "soyos",
    "habréis",
    "enfrente",
    "empleo",
    "si",
    "ésos",
    "aquélla",
    "aseguró",
    "buena",
    "ex",
    "tenéis",
    "mencionó",
    "sus",
    "ir",
    "señaló",
    "arribaabajo",
    "cuales",
    "da",
    "ver",
    "debajo",
    "podeis",
    "eras",
    "ante",
    "éstas",
    "casi",
    "contra",
    "existe",
    "parece",
    "tenemos",
    "fuiste",
    "hube",
    "existen",
    "habido",
    "tuvieses",
    "quedó",
    "todo",
    "ninguno",
    "hay",
    "sea",
    "estais",
    "eso",
    "ahí",
    "pocas",
    "hemos",
    "temprano",
    "expresó",
    "muchos",
    "hubimos",
    "pronto",
    "hubo",
    "solamente",
    "ha",
    "teneis",
    "cuenta",
    "estuvimos",
    "estando",
    "h",
    "segunda",
    "suyas",
    "fue",
    "dice",
    "tendrás",
    "valor",
    "hubieseis",
    "seis",
    "incluso",
    "q",
    "serás",
    "atras",
    "sera",
    "habían",
    "suyo",
    "habré",
    "o",
    "estado",
    "cuánto",
    "sois",
    "fuese",
    "tenida",
    "diferente",
    "podria",
    "hayamos",
    "aunque",
    "sabeis",
    "nuestro",
    "quizas",
    "tener",
    "estas",
    "consiguen",
    "tuve",
    "trabajamos",
    "ambos",
    "les",
    "solas",
    "g",
    "ampleamos",
    "antaño",
    "anterior",
    "propios",
    "consideró",
    "despacio",
    "tuvieseis",
    "poner",
    "posible",
    "medio",
    "le",
    "tampoco",
    "junto",
    "asi",
    "vaya",
    "primer",
    "cuántas",
    "ayer",
    "fui",
    "cuanta",
    "cuántos",
    "tenía",
    "cuantas",
    "hasta",
    "respecto",
    "demasiado",
    "la",
    "varias",
    "luego",
    "debe",
    "tenidas",
    "también",
    "dias",
    "otros",
    "modo",
    "primeros",
    "d",
    "mio",
    "vez",
    "segundo",
    "estuve",
    "dos",
    "tuvieron",
    "ahora",
    "saber",
    "dónde",
    "tened",
    "tres",
    "mías",
    "al",
    "alrededor",
    "todos",
    "sin",
    "ni",
    "dieron",
    "llevar",
    "seríais",
    "sé",
    "ti",
    "trabajas",
    "todas",
    "antano",
    "habrían",
    "hablan",
    "hubieras",
    "hacia",
    "emplear",
    "habremos",
    "hubiéramos",
    "dijo",
    "excepto",
    "mío",
    "sido",
    "cerca",
    "cierto",
    "quién",
    "hubieran",
    "intento",
    "tenidos",
    "estéis",
    "ese",
    "n",
    "lo",
    "tuviera",
    "bueno",
    "intentas",
    "habia",
    "comentó",
    "manifestó",
    "por",
    "quien",
    "algo",
    "estarán",
    "parte",
    "sería",
    "haciendo",
    "habríamos",
    "estarás",
    "cuáles",
    "habrán",
    "realizar",
    "tenido",
    "intentan",
    "sigue",
    "estaríamos",
    "fuésemos",
    "ninguna",
    "añadió",
    "menudo",
    "tuvisteis",
    "sino",
    "mia",
    "fuisteis",
    "estados",
    "tercera",
    "eres",
    "estás",
    "tarde",
    "raras",
    "estuvieras",
    "uno",
    "dejó",
    "tengan",
    "habías",
    "van",
    "vosotras",
    "estabais",
    "ahi",
    "nuevos",
    "sabe",
    "seáis",
    "dar",
    "nosotras",
    "estamos",
    "trabajo",
    "estar",
    "tuviste",
    "tenías",
    "alguna",
    "considera",
    "yo",
    "consigues",
    "tuvimos",
    "están",
    "mucha",
    "eramos",
    "pueda",
    "fuéramos",
    "nosotros",
    "creo",
    "esas",
    "tendré",
    "nuestros",
    "este",
    "principalmente",
    "usar",
    "ya",
    "aproximadamente",
    "poca",
    "dentro",
    "tendremos",
    "dijeron",
    "habiendo",
    "aquel",
    "día",
    "entonces",
    "entre",
    "f",
    "fin",
    "mí",
    "trabajais",
    "salvo",
    "tuvierais",
    "solo",
    "trata",
    "intentamos",
    "buen",
    "buenas",
    "usais",
    "últimos",
    "después",
    "primero",
    "tuviésemos",
    "verdadero",
    "conmigo",
    "otra",
    "trabajar",
    "serían",
    "estará",
    "ella",
    "estuvieses",
    "según",
    "vais",
    "aquellos",
    "tuviese",
    "próximo",
    "esos",
    "peor",
    "diferentes",
    "de",
    "menos",
    "buenos",
    "del",
    "suya",
    "aquél",
    "algún",
    "teniendo",
    "podriamos",
    "veces",
    "vosotros",
    "éste",
    "antes",
    "estuviésemos",
    "horas",
    "dicen",
    "igual",
    "erais",
    "realizó",
    "habéis",
    "hubiésemos",
    "mias",
    "más",
    "segun",
    "fueras",
    "b",
    "cierta",
    "vuestra",
    "con",
    "quiza",
    "solos",
    "teníais",
    "habida",
    "enseguida",
    "tiene",
    "fuesen",
    "sal",
    "tuvieran",
    "tendrá",
    "esta",
    "fueron",
    "estad",
    "total",
    "misma",
    "donde",
    "esa",
    "aquí",
    "pocos",
    "estos",
    "a",
    "tuviesen",
    "vuestro",
    "dio",
    "paìs",
    "soy",
    "haber",
    "seremos",
    "va",
    "somos",
    "dia",
    "intenta",
    "conocer",
    "lleva",
    "hayas",
    "os",
    "podrían",
    "habrás",
    "él",
    "tuvo",
    "estoy",
    "cosas",
    "repente",
    "mía",
    "conseguir",
    "usas",
    "hayáis",
    "tanto",
    "tienen",
    "pasada",
    "aquéllos",
    "estarías",
    "mejor",
    "desde",
    "seríamos",
    "tuvieras",
    "cuándo",
    "informo",
    "próximos",
    "era",
    "estuvierais",
    "estés",
    "seas",
    "puede",
    "empleais",
    "hubieses",
    "hubiesen",
    "podemos",
    "serías",
    "detras",
    "durante",
    "podrán",
    "cuantos",
    "mismos",
    "ésas",
    "pasado",
    "sean",
    "vamos",
    "cada",
    "habrá",
    "las",
    "haya",
    "estuvieran",
    "hubieron",
    "sabes",
    "tengáis",
    "pueden",
    "consigo",
    "podrias",
    "tenían",
    "cualquier",
    "hubiste",
    "vuestros",
    "ciertas",
    "pesar",
    "xq",
    "había",
    "tiempo",
    "ningún",
    "serán",
    "adelante",
    "gran",
    "esto",
    "ademas",
    "verdad",
    "verdadera",
    "cuál",
    "tengamos",
    "c",
    "pudo",
    "esté",
    "cual",
    "estada",
    "mismo",
    "ningunas",
    "sí",
    "debido",
    "estuvieron",
    "podrian",
    "estén",
    "partir",
    "poder",
    "estaban",
    "tan",
    "dado",
    "cuanto",
    "se",
    "ustedes",
    "nueva",
    "el",
    "otras",
    "ejemplo",
    "llegó",
    "usted",
    "tuyos",
    "lado",
    "fueses",
    "podría",
    "alli",
    "sobre",
    "mismas",
    "nos",
    "embargo",
    "ultimo",
    "teníamos",
    "tras",
    "que",
    "hubisteis",
    "habría",
    "supuesto",
    "cuatro",
    "habla",
    "encima",
    "estaríais",
    "contigo",
    "estuviera",
    "tendrían",
    "habíais",
    "tuviéramos",
    "usan",
    "adrede",
    "estarían",
    "quiere",
    "un",
    "informó",
    "demás",
    "habríais",
    "algunas",
    "hizo",
    "pero",
    "tuyas",
    "v",
    "deben",
    "así",
    "saben",
    "ser",
    "te",
    "nuevo",
    "consigue",
    "hayan",
    "bastante",
    "siete",
    "fuerais",
    "estuviéramos",
    "tal",
    "qeu",
    "siguiente",
    "propias",
    "trabajan",
    "cuánta",
    "algunos",
    "explicó",
    "haceis",
    "quienes",
    "realizado",
    "habidos",
    "en",
    "puedo",
    "siendo",
    "estuviese",
    "final",
    "alguno",
    "unas",
    "fuimos",
    "ésa",
    "ojalá",
]

extended_stopwords = list(set(default_stopwords + extended_stopwords))

punct = [
    ".",
    ",",
    ":",
    ";",
    "!",
    "?",
    "¿",
    "¡",
    "(",
    ")",
    "[",
    "]",
    "{",
    "}",
    "/",
    "\\",
    "|",
    "-",
    "_",
    "–",
    "—",
    "…",
    "·",
    "•",
    "°",
    "´",
    "`",
    "'",
    '"',
    "“",
    "”",
    "‘",
    "’",
    "«",
    "»",
    "‹",
    "›",
]