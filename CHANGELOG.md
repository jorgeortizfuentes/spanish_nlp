## [0.5.0] - 2026-10-03

### 🐛 Bug Fixes

- Detect hashtags at beginning of text - ([962fe84](https://github.com/jorgeortizfuentes/spanish_nlp/commit/962fe841f4eabc68d4c0984ab49f7a95c70f1b08))
- Resolve ruff lint findings - ([5fface9](https://github.com/jorgeortizfuentes/spanish_nlp/commit/5fface91b5a89d495c23954309adab1c251d5403))
- Correct all and character removal spelling augmentations - ([63c7888](https://github.com/jorgeortizfuentes/spanish_nlp/commit/63c7888d6d04fcc8159992eb12fee11c8a016062))
- Rebuild text chunks from wordpiece tokens in masked augmentation - ([01b40fd](https://github.com/jorgeortizfuentes/spanish_nlp/commit/01b40fdbccfb69fd644651dfea0371804e7fa403))
- Raise valueerror for unsupported augment input types - ([6c3e051](https://github.com/jorgeortizfuentes/spanish_nlp/commit/6c3e051d70a04ea3729ce49b8ec895abc93a0962))

### 💼 Other

- Migrate dependency management to uv - ([4ffb26a](https://github.com/jorgeortizfuentes/spanish_nlp/commit/4ffb26ad1eea4d0ad00377eccad8f2a29f9cbbea))
- Use uv commands in makefile - ([3c06b6a](https://github.com/jorgeortizfuentes/spanish_nlp/commit/3c06b6a34c9c6da732dbac762b36e5f6d7bf32f8))
- Configure ruff lint rules - ([ca33fa8](https://github.com/jorgeortizfuentes/spanish_nlp/commit/ca33fa83f446f147a2fc22b43204736127083a1a))
- Upgrade packages - ([f1a7aa5](https://github.com/jorgeortizfuentes/spanish_nlp/commit/f1a7aa56705a7fcab6f66099771c9bcfebe0c4f5))

### 📚 Documentation

- Document uv development workflow - ([911673f](https://github.com/jorgeortizfuentes/spanish_nlp/commit/911673f3adb834a276fd4aaaa08e75911ee7baea))
- Adds changelog with cliff.toml - ([5125344](https://github.com/jorgeortizfuentes/spanish_nlp/commit/512534486051142333d1a8d0aadb8f8b93076275))

### ⚡ Performance

- Load spacy tokenizer once per process - ([1208f04](https://github.com/jorgeortizfuentes/spanish_nlp/commit/1208f04723ad88b2e9a83559aae4498d7a43c3ff))

### 🎨 Styling

- Uses ruff to format code ([#12](https://github.com/jorgeortizfuentes/spanish_nlp/issues/12)) - ([b813b86](https://github.com/jorgeortizfuentes/spanish_nlp/commit/b813b86a1e0089751868fdfffc485f899cc277da))
- Apply ruff autofixes and format notebooks - ([6edc1db](https://github.com/jorgeortizfuentes/spanish_nlp/commit/6edc1db1203ae42c4b29d10150e097a16fd3414e))

### 🧪 Testing

- Create outputs directory before running tests - ([ad8aa8e](https://github.com/jorgeortizfuentes/spanish_nlp/commit/ad8aa8ecc63a2b58417d76c17c5ed7821741d086))
- Rewrite augmentation tests with faster, stricter checks - ([d1a125c](https://github.com/jorgeortizfuentes/spanish_nlp/commit/d1a125c60c4fa13c52fe87aa7e6c46d64274fee4))
- Remove empty classifier test and unused outputs fixture - ([bdac038](https://github.com/jorgeortizfuentes/spanish_nlp/commit/bdac038f97a8cbbeab878fcce6803322217c2fc7))

### ⚙️ Miscellaneous Tasks

- Adds ruff and pytest as dev dependencies to pyproject.toml - ([a70d949](https://github.com/jorgeortizfuentes/spanish_nlp/commit/a70d9495c8cd40b565fcf101e94eaa168a194779))
- Adds workflow for ruff check ([#12](https://github.com/jorgeortizfuentes/spanish_nlp/issues/12)) - ([bf56642](https://github.com/jorgeortizfuentes/spanish_nlp/commit/bf566426a8b80249da1a98636ff9cfe55c003eed))
- Use uv in github actions workflow - ([9437944](https://github.com/jorgeortizfuentes/spanish_nlp/commit/94379449ea1ec2f6856fa0fdde75f5a1cff24635))
- Run ruff with the locked version - ([94f00d5](https://github.com/jorgeortizfuentes/spanish_nlp/commit/94f00d5320480c32b252f4b57c201e89b00b14ca))
- Test on Python 3.10-3.12 and split deploy job - ([63fb1c6](https://github.com/jorgeortizfuentes/spanish_nlp/commit/63fb1c62fa3cada39b94bde39164cd7dc0c0e2fa))
- Adds cliff.toml as changelog generator config ([#17](https://github.com/jorgeortizfuentes/spanish_nlp/issues/17)) - ([d96fef2](https://github.com/jorgeortizfuentes/spanish_nlp/commit/d96fef2438949c543474dde9ebed7e00bddaa802))


## [0.4.0] - 2025-04-26

### 🚀 Features

- Add spell checker with dictionary and contextual LM backends - ([117350c](https://github.com/jorgeortizfuentes/spanish_nlp/commit/117350caf75bce6e7fcf8277099fa98ff64e1b14))
- Refactor contextual checker for hybrid LM/dictionary correction - ([5ad9496](https://github.com/jorgeortizfuentes/spanish_nlp/commit/5ad9496d695f11fed26587393c5fa28a8d921816))
- Add skeleton for ContextualLMSpellChecker - ([23d3087](https://github.com/jorgeortizfuentes/spanish_nlp/commit/23d3087e14a2b57f5f8fa12d9a6e45f3fbae2713))

### 🐛 Bug Fixes

- Update dependencies to resolve spacy/pydantic error - ([d7bf166](https://github.com/jorgeortizfuentes/spanish_nlp/commit/d7bf166736396b2f9ad6025df704a40f3554946d))
- Handle None returned by pyspellchecker candidates - ([781bde7](https://github.com/jorgeortizfuentes/spanish_nlp/commit/781bde7e66028d650cf48e507f254f8b94e360de))
- Remove optimization skipping LM for dict-correct words - ([6dd4596](https://github.com/jorgeortizfuentes/spanish_nlp/commit/6dd4596d53637024227ec67f61d547a412dd7255))
- Adjust spellchecker tests to match pyspellchecker behavior - ([de0a712](https://github.com/jorgeortizfuentes/spanish_nlp/commit/de0a71249887024e7c96be993cd64ae67fcd44e8))

### 💼 Other

- Update dependencies versions in requirements.txt - ([16bcf60](https://github.com/jorgeortizfuentes/spanish_nlp/commit/16bcf60a2701958ab860d7372d1979ddc75aa14a))
- Update spellchecking notebook outputs text examples markdown and execution counts - ([b5da6c5](https://github.com/jorgeortizfuentes/spanish_nlp/commit/b5da6c540b217fe3987addd64b9c4f6e2370fc45))
- Remove contextual language model example from spellchecking notebook - ([06bb4fe](https://github.com/jorgeortizfuentes/spanish_nlp/commit/06bb4fe94d3f341aea5818a42df751bcbf236b25))
- Update readme - ([47aef9a](https://github.com/jorgeortizfuentes/spanish_nlp/commit/47aef9a274d7f1ee05aa996cde3143417f25f34f))
- Remove references to sphinx and bump2version from development conventions - ([d618224](https://github.com/jorgeortizfuentes/spanish_nlp/commit/d6182245b4913c52c860ef8e2def6e18f20b87ee))
- Bump version to 0.4.0 - ([3687d4d](https://github.com/jorgeortizfuentes/spanish_nlp/commit/3687d4db779754f85d4518ef30fbb4217e3fe6ba))
- Update development dependencies in requirements-dev.txt - ([bb3eaf1](https://github.com/jorgeortizfuentes/spanish_nlp/commit/bb3eaf17065db636f0a2240df9ecc15d7d4ea52e))

### 🚜 Refactor

- Remove non-functional contextual_lm spellchecker - ([f85fd5e](https://github.com/jorgeortizfuentes/spanish_nlp/commit/f85fd5edc54b42456ec7df0f6ce41823136b9ef1))

### 📚 Documentation

- Execute Spellchecking notebook and save outputs - ([cba5eae](https://github.com/jorgeortizfuentes/spanish_nlp/commit/cba5eae0f0e23df0e82b3e62d8009fb50639167f))
- Separate examples for custom distance and dictionary in notebook - ([c1e5427](https://github.com/jorgeortizfuentes/spanish_nlp/commit/c1e54273c2d684964a46343d384da800e714dc46))
- Update custom dictionary example in Spellchecking notebook - ([db69e64](https://github.com/jorgeortizfuentes/spanish_nlp/commit/db69e647713943078bb756312383a95405a599a5))
- Update spellchecking example notebook - ([edbc849](https://github.com/jorgeortizfuentes/spanish_nlp/commit/edbc849cf8eaa9aae6c4d03f12e4c714df8c378c))
- Add spell checker documentation to README - ([6d61eca](https://github.com/jorgeortizfuentes/spanish_nlp/commit/6d61eca2fac0c0800e309821645315ebc33345dd))
- Update table of contents - ([a96f478](https://github.com/jorgeortizfuentes/spanish_nlp/commit/a96f478c7001f4021edd83eef41fa7df145bc59b))
- Add developer guide - ([e01a59a](https://github.com/jorgeortizfuentes/spanish_nlp/commit/e01a59a9c1aaf81b9848112aeb9c618febe2dddd))
- Document Gitflow contribution workflow in README-dev - ([ffec2de](https://github.com/jorgeortizfuentes/spanish_nlp/commit/ffec2debe02b3d66529b9e96e81c597323f9416f))
- Clarify Gitflow process and main branch protection - ([dace7a0](https://github.com/jorgeortizfuentes/spanish_nlp/commit/dace7a0152939a7ffcfa199300bc0ce01e24e147))
- Clarify Gitflow workflow and PR process in CONTRIBUTING - ([040f58b](https://github.com/jorgeortizfuentes/spanish_nlp/commit/040f58b445ee386a1d459ba8fe5a4de8306cfe7f))
- Add reference to CONTRIBUTING.md in README - ([07942e5](https://github.com/jorgeortizfuentes/spanish_nlp/commit/07942e54e1b378fcd262616346d511e61527ff60))
- Add reference to CONVENTIONS.md in CONTRIBUTING.md - ([6d1009e](https://github.com/jorgeortizfuentes/spanish_nlp/commit/6d1009e827e0a1af44de67c7cbe58c28cc3a525e))
- Improve formatting and update instructions in contributing guide - ([7011e8a](https://github.com/jorgeortizfuentes/spanish_nlp/commit/7011e8af90e774d594aa30eeecee2e039a859099))
- Add PyPI downloads badge to README - ([afa0256](https://github.com/jorgeortizfuentes/spanish_nlp/commit/afa025603696f5c5064d9f0b3009a4766a56b55f))
- Add maintenance and license badges to README - ([a803191](https://github.com/jorgeortizfuentes/spanish_nlp/commit/a8031912586c7c7a0b4a924f9f316a9ebbdd8176))

### 🧪 Testing

- Add tests for dictionary spell checker - ([9b56796](https://github.com/jorgeortizfuentes/spanish_nlp/commit/9b56796b8b96ab461a791990ea4a1861e60bde90))
- Fix spellchecker tests to match dictionary behavior - ([155b5fe](https://github.com/jorgeortizfuentes/spanish_nlp/commit/155b5fec386a16ab06f6866eeec7901c6aa2394f))

### ⚙️ Miscellaneous Tasks

- Rename README-dev.md to CONTRIBUTING.md - ([bc0dd75](https://github.com/jorgeortizfuentes/spanish_nlp/commit/bc0dd75789fcb3722acad17b0a5bd1596b2ae58e))
- Trigger workflow only on main branch push - ([9584816](https://github.com/jorgeortizfuentes/spanish_nlp/commit/9584816dc1a68b5f8cf3824746f1ef1ab9c03760))
- Run tests on develop branch - ([06a4b79](https://github.com/jorgeortizfuentes/spanish_nlp/commit/06a4b79bd4d7d61bca961510831c0c322b7f1f39))
- Run tests on feature branches, build only on main - ([2e8ca7a](https://github.com/jorgeortizfuentes/spanish_nlp/commit/2e8ca7a764d84e1dfedfc27ddea35c95817cae87))
- Remove comments from main.yml - ([55c5cdd](https://github.com/jorgeortizfuentes/spanish_nlp/commit/55c5cddd1064d6dd052cc2d54faac576ac2c92d6))


## [0.3.1] - 2025-01-20

### 💼 Other

- Requires python version - ([f80c9c4](https://github.com/jorgeortizfuentes/spanish_nlp/commit/f80c9c43a13d8ea055cc7abe9f8473d5dd0f51a4))

### 📚 Documentation

- Enhance README for better presentation and clarity - ([9de936a](https://github.com/jorgeortizfuentes/spanish_nlp/commit/9de936af867bac2b1f6f5cca1aebd425b8d4d6d4))
- Update README structure and contact details - ([58cb7fb](https://github.com/jorgeortizfuentes/spanish_nlp/commit/58cb7fbe48288da55b5c440afe601af41ce1e2f9))
- Update README formatting and installation info - ([13ba3ff](https://github.com/jorgeortizfuentes/spanish_nlp/commit/13ba3ffe3e459a96c557a3ca27e2215209df46e6))

### ⚙️ Miscellaneous Tasks

- Bump version to 0.3.1 - ([9a6ef73](https://github.com/jorgeortizfuentes/spanish_nlp/commit/9a6ef7350d8271f60944e4f8de4f492cee37c3d1))


## [0.3.0] - 2025-01-19

### 🚀 Features

- Conventions and Makefile - ([ec76fef](https://github.com/jorgeortizfuentes/spanish_nlp/commit/ec76fefb63b8baf7cb2e00256d84e93f3241ccd0))
- Add ipywidgets to dev requirements - ([1445af3](https://github.com/jorgeortizfuentes/spanish_nlp/commit/1445af3301970cfabfa4e698acaadc8b49757cc1))
- Add pytest target to Makefile - ([4533e43](https://github.com/jorgeortizfuentes/spanish_nlp/commit/4533e43d648d60d0a62c9155261df72212f5238b))
- Save test outputs to outputs dir and add coverage - ([a5a210d](https://github.com/jorgeortizfuentes/spanish_nlp/commit/a5a210dc8b42c5b0c179d9ce52ccf2761af541bd))
- Add uv installation to GitHub Actions workflow - ([a0a40fe](https://github.com/jorgeortizfuentes/spanish_nlp/commit/a0a40fef8c0b7cdfabde8e419361068c39a9dcf7))

### 🐛 Bug Fixes

- Update init to include preprocess module - ([c6031e0](https://github.com/jorgeortizfuentes/spanish_nlp/commit/c6031e0d54b5b882e6e224db299618155b297614))
- Import SpanishPreprocess in test file - ([8750da0](https://github.com/jorgeortizfuentes/spanish_nlp/commit/8750da0641752cb8e0d1ebfaa53a93d576c40d60))
- Activate virtual environment before build - ([4129076](https://github.com/jorgeortizfuentes/spanish_nlp/commit/41290762f92a58b19c2b5043fd580e07d77c930d))
- Adjust CI workflow for editable install and build - ([61d2561](https://github.com/jorgeortizfuentes/spanish_nlp/commit/61d25610814f0f96a226d3fa4815a7ee28dcedf5))

### 💼 Other

- Requirements - ([2c4dce1](https://github.com/jorgeortizfuentes/spanish_nlp/commit/2c4dce132a5d785ab36afc6530ddeaf2be73b5fc))
- Requirements-dev to makefile - ([0766d2e](https://github.com/jorgeortizfuentes/spanish_nlp/commit/0766d2ebf3e06021cf743954123a8f7328aaa8cb))
- Requirements-dev to makefile and set python version - ([6a4edf2](https://github.com/jorgeortizfuentes/spanish_nlp/commit/6a4edf27e2d695caa3cb066227b7e8195e21805b))
- Numpy 1.26.4 - ([52cc371](https://github.com/jorgeortizfuentes/spanish_nlp/commit/52cc37197f26da28d4d15efd897314194d72e38b))
- Move classifiers to a folder - ([f21a747](https://github.com/jorgeortizfuentes/spanish_nlp/commit/f21a747693c7cbb13e8aa21065ca622879662c94))
- Refactor classifiers code - ([b0b4e88](https://github.com/jorgeortizfuentes/spanish_nlp/commit/b0b4e88e7b9bdf1e60a6d58a4dedca397343b111))
- Swifter - ([3329419](https://github.com/jorgeortizfuentes/spanish_nlp/commit/3329419f8ed202fe8e4f5fe42062ce18dbe65b0c))
- Outputs to gitignore - ([c05c34b](https://github.com/jorgeortizfuentes/spanish_nlp/commit/c05c34b5337111b624847f6c00ff060a274fd867))
- Run new examples - ([f73e117](https://github.com/jorgeortizfuentes/spanish_nlp/commit/f73e117d463a7d84e64d4147de62aaea68f37442))
- Version to 0.3.0 - ([c706d89](https://github.com/jorgeortizfuentes/spanish_nlp/commit/c706d898303983d1aa2cc8e2633df1780eb8da1b))
- Create venv in workflow - ([0453580](https://github.com/jorgeortizfuentes/spanish_nlp/commit/045358041b03e099ff242652455947c530c0d98e))

### 🚜 Refactor

- Preprocess - ([aa597b7](https://github.com/jorgeortizfuentes/spanish_nlp/commit/aa597b7e0b5f1683d0296eae68ff57a1d43791a6))
- Use logging instead of print in transform method - ([8984320](https://github.com/jorgeortizfuentes/spanish_nlp/commit/8984320bff16be3f9a5c528f9a2feabe053b891f))
- Use Makefile for running tests - ([6c87740](https://github.com/jorgeortizfuentes/spanish_nlp/commit/6c877405972db1dc883de73a0f9671e702dd8347))
- Simplify dependency installation in CI workflow - ([8314b89](https://github.com/jorgeortizfuentes/spanish_nlp/commit/8314b89ddf735f6bcfc39413a80276ff5219703d))

### ⚙️ Miscellaneous Tasks

- Use Makefile for dependency installation in CI - ([1eb4e07](https://github.com/jorgeortizfuentes/spanish_nlp/commit/1eb4e07f373444e159415f1a53a793c25206e134))
- Run workflow on develop and feature branches, publish on main - ([48f4935](https://github.com/jorgeortizfuentes/spanish_nlp/commit/48f493571ebee05b85cbff27f8a938bdc69eb97e))


## [0.2.1] - 2023-02-26


