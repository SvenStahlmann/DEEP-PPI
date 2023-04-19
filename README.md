<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/SvenStahlmann/DEEP-PPI">
    <img src="DNA_helix_round.png" alt="Logo" width="340" height="180">
  </a>

<h3 align="center">DEEP-PPI</h3>

  <p align="center">
    Protein-Protein Interaction using deep learning models 
    <br />
  </p>
</div>

<!-- ABOUT THE PROJECT -->
## About The Project

This project aims to develop and train deep learning models to predict the interaction of proteins by using protein sequences as inputs. 
The goal is to provide accurate predictions that can be used to study the complex interactions between proteins. 
The models are based on large Language Models as encoders.
If you find this reseach interesting and have questions please reach out to me (e.g. through github issues).

## Getting Started

This project is based on Python 3.11 and built using [Poetry](https://python-poetry.org/).
### Prerequisites

Please install Poetry and clone the github repository. Poetry will handle the creation of a vitual enviroment and the installation of all dependencies.

### Installation

To install the dependencies using poetry 
`
poetry install
`
in the directory of the repository.

You can activate the virtual env using `poetry shell`. If you want to use GPU acceleration please install [pytorch](https://pytorch.org/get-started/locally/ ) in this enviroment.

## Usage

After installation use `poetry run python main.py` to run the main.py file.

## Models

### BaseLineModel

The model uses a pretrained encoder model (ESM) to transform the two protein sequences into embeddings. The resulting embeddings are concatinated and fed into a head. The head is a 2 layer deep fully connected network. Softmax is applied to the logits of the head to generate the probabilites of interaction between the two input proteins.



