# AudioSet-Tools
This repository provides code and examples for "S. Giacomelli et al. - AudioSet-Tools: a Python Research Framework for Custom AudioSet Distributing and Processing" (LINK_ToDo).

[![AudioSet](https://production-media.paperswithcode.com/datasets/Screen_Shot_2021-01-28_at_9.31.55_PM.png)](https://research.google.com/audioset/download.html)

[![python](https://img.shields.io/badge/Python-3.11-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![jupyter](https://img.shields.io/badge/Jupyter-Lab-F37626.svg?style=flat&logo=Jupyter)](https://jupyterlab.readthedocs.io/en/stable)
[![pytorch](https://img.shields.io/badge/PyTorch-2.6.0-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)

AudioSet-Tools is a modular Python framework for filtering, downloading, and post-processing extensive audio collections derived from Google’s AudioSet. This framework streamlines label-based selection, dataset balancing, and automated download and DSP pipelines, enabling end-to-end and reproducible research workflows based on [AudioSet](https://research.google.com/audioset/).

## How to Use

Clone this repository locally:

```bash
git clone https://github.com/StefanoGiacomelli/audioset-tools.git
```

the run:

```sh
cd ./audioset-tools/
pip install -r requirements.txt
```

## Project structure
    ./
    ├── AudioSet_EV_data/                   
    |   ├── EV_Negatives.xxx        # Negatives group files (original dataset .csv, post-download .csv and statistics .json)
    |   ├── EV_Positives.xxx        # Positives group files (original dataset .csv, post-download .csv and statistics .json)
    |
    ├── audioset_tools/
    |   ├── downloaders.py          # it contains AudioSet downloading class and functions
    |   ├── filters.py              # it contains AudioSet .csv filtering functions
    |   ├── utils.py                # it contains AudioSet .csv utility functions
    |   ├── original_csv/           # it contains a pre-downloaded AudioSet .csv distribution (dated 01-11-2024)
    |       ├── ...
    |
    ├── EV-benchmark/               # benchmarking datasets folder for Emergency Vehicle recognition
    |   ├── ...                     # dataset-specifc folder: contains a 'ReadMe.md' to guide through contents download and set-up
    |   ├── dataloaders.py          # it contains all PyTorch (Lightning) benchmarks Dataset and DataModule implementations 
    |   ├── data_demo.py            # a Python script to showcase benchmark usage (statistics extraction)
    |
    ├── main_ev_processing.py       # AudioSet-EV .csv processing pipeline (it serves as both doc and reference)
    ├── main_download.py            # AudioSet-EV downloading script (it serves as both doc and reference)
    ├── requirements.txt            
    ├── LICENSE
    └── README


## Release History

* 0.1.0
    * Official public release: IEEE paper submission
* 0.2.0 [PLANNED... --> Open to Contributions!]
    * CHANGE: compile functions and class documentations on ReadTheDocs
    * CHANGE: compile audioset-tools as a Python package and host on PyPI

## Meta

Stefano Giacomelli – DISIM dpt. (University of L'Aquila, ITA) – stefano.giacomelli@graduate.univaq.it
Distributed under the GPL license. See ``LICENSE`` for more information.

https://github.com/StefanoGiacomelli

## Contributing

1. Fork it (<https://github.com/StefanoGiacomelli/audioset-tools/fork>)
2. Create your feature branch (`git checkout -b feature/new_feature`)
3. Commit your changes (`git commit -am 'Add new_features'`)
4. Push to the branch (`git push origin feature/new_features`)
5. Create a new Pull Request (Thank you in advance!)

# Reference

If you find this repository or AudioSet-EV useful, please cite this:

```
ref_ToDo
```

