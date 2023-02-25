from setuptools import find_packages, setup

with open("README.md", "r", encoding = "utf-8") as fh:
    long_description = fh.read()

setup(
    name="spanish_nlp",
    version="0.1.7",
    description="A package for NLP in Spanish",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url="https://github.com/jorgeortizfuentes/spanish_nlp",
    author="Jorge Ortiz-Fuentes",
    author_email="jorge@ortizfuentes.com",
    license="GNU General Public License v3.0",
    # packages=["spanish_nlp", "spanish_nlp.utils", "spanish_nlp.augmentation"],
    install_requires=[
        "pandas",
        "emoji",
        "nltk",
        "torch",
        "transformers",
        "datasets",
        "swifter",
        "kaleido",
        "es_core_news_sm",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Text Processing :: Linguistic",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    package_dir = {"": "src"},
    packages=find_packages("src"),
    keywords="nlp pln spanish español classifier clasificacion augmentation aumento preprocess preprocesamiento",
    python_requires=">=3.6",
)
