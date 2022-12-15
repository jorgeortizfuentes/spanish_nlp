from setuptools import setup

setup(
    name='Spanish NLP',
    version='0.1.3',
    description='A package for Spanish NLP',
    url='https://github.com/jorgeortizfuentes/spanish_nlp',
    author='Jorge Ortiz Fuentes',
    author_email='jorge@ortizfuentes.com',
    license='GNU General Public License v3.0',
    packages=['spanish_nlp', 'spanish_nlp.utils'],
    install_requires=['pandas',
                      'emoji',
                      'nltk',
                      'transformers',
                      'swifter',
                      'kaleido',      
                      ],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Topic :: Text Processing :: Linguistic',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3.9',
    ],
    keywords='nlp spanish',
    python_requires='>=3.6',
)
