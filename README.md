# Pretty Plain Readme File

[![Python 3.12.4](https://img.shields.io/badge/python-3.12.4-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Repo Size](https://img.shields.io/github/repo-size/mgrube753/AS-SS2026-Oil-Slick-Detection.svg)](https://github.com/mgrube753/AS-SS2026-Oil-Slick-Detection)
[![University of Rostock](https://img.shields.io/badge/Institution-University_of_Rostock-003D7A.svg)](https://www.uni-rostock.de/)

What's needed?

- `requirements.txt` for dependencies
- `README.md` filled with project description and instructions

## Set Up

1. Clone the repository

    ```bash
    git clone https://github.com/mgrube753/AS-SS2026-Oil-Slick-Detection.git
    cd AS-SS2026-Oil-Slick-Detection
    ```

2. Create a virtual environment (recommended):
   _I am using pyenv for managing several Python versions for different projects. You can also use `venv` or `conda` if you prefer. Best practise: using Python 3.12.4 together._

    venv example:

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3. In your environment, install dependencies

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. While being in your environment, download the OilSlick dataset subset from Hugging Face by running the `download_data.sh` script from the repository root directory:

    ```bash
    ./download_data.sh
    ```

    The progress bar updates are feeling very delayed, but the download is actually progressing. Please be patient.

2. Then, run the `extract_data.sh` script to unpack the dataset properly:

    ```bash
    ./extract_data.sh
    ```

That's all till now. :)
