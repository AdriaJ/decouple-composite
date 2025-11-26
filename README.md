Repo content
============

This repo contains the code related to the article [Decoupled Solution for Composite Sparse-plus-Smooth Inverse Problems](https://arxiv.org/abs/2510.23322).

It performs simple composite reconstruction in the setup detailed in the article.
It also contains the code to perform a batch of experiments and synthesize the dataset analyzed for comparing the performance of decoupled reconstruction in different setups.

The dataset produced while writing the article is saved on [Zenodo]() [TODO: upload the dataset and link it here !].

File description
----------------

Description of the files and how to use them.

**Simple reconstructions:**
- `composite_reconstruction.py`
  : Perform composite reconstruction with simple quadratic Hilbert-norm penalty.

- `rkhs_composite.py`
  : Composite reconstruction with quadratic RKHS-norm penalty, to promote smoother Hilbert-space solutions.

**Database generation:**
- `make_db_l2.py`
  : Generate a database of reconstructions, within the framework of TV + L2 penalties.

- `make_db_rkhs.py`
  : Generate a database of reconstructions, within the framework of TV + RKHS penalties.

- `db_files/`
  : Various files used used in the generation of the databases.

**Result analysis:**
- `open_db.py`
  : Open a database generated with one of the above-mentioned scripts and generates the analysis figures as in the article.

- `plot_case.py`
  : Open a single case from a database and draw the reconstructions, used to illustrate a solved example.

**Miscellanious:**
- `utils.py`
  : Utility functions, simply compute error metrics.

- `draw_measurement_fig.py`
  : Draw the figure used in the article to illustrate the measurement procedure.

- `scratch/`
  : Various notes, code and debugging files.

How to install
--------------

Python environment is provided, run the following instructions to set it up.

```bash
$ conda env create -f conda-env.yml
$ conda activate decoupling
$ pip install -r pip-requirements.txt
```

