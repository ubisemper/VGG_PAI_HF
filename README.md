# Project Title

This project contains scripts for training VGG models for the high frequency ultrasound imaging dataset, and the PAI_SIM_V2 dataset


## TODO

This script trains a VGG model using a custom `NPZDataset`. It uses wandb for logging and argparse for command line arguments. The script requires the same arguments as `vgg_trainer_HF.py`, with the addition of `model_path`.

## How to Run

To run the scripts, use the following command:

```sh
python <script_name> --project_name <project_name> --run_name <run_name> --lr <lr> --momentum <momentum> --batch_size <batch_size> --epochs <epochs> --freeze_proportion <freeze_proportion>