# PAR: Poly-Autoregressive Prediction for Modeling Interactions

[Neerja Thakkar](neerja.me), [Tara Sadjadpour](https://github.com/tsadja), [Jathushan Rajasegaran](https://brjathu.github.io/), [Shiry Ginosar](https://shiry.ttic.edu/), [Jitendra Malik](https://people.eecs.berkeley.edu/~malik/)

[[`Paper`](https://arxiv.org/abs/2502.08646)] [[`Project`](https://neerja.me/PAR/)]

## AVA Action Prediction

Training the models requires downloading the AVA dataset, and using the visualization code requires having the AVA videos downloaded.

To setup the environment for this task, run the following command. Note that the $HOME path should be updated for the `prefix` in the environment's yaml file.
```
conda env create -f ava_actions.yml
```

This code trains on AVA GT action labels. It takes trajectories of action labels at 1Hz (one set of labels per agent per second). It predicts past action labels from the future (at test time, it takes half of the video length's actions as the past input and predicts the second half).

The key files for each task are as follows, with examples for this task:
- Dataloader: `PAR_v2/lart/datamodules/components/ava_dataset.py`, Datamodule `PAR_v2/lart/datamodules/ava_datamodule.py``
- Config: `PAR_v2/configs/ava.yaml`
- Training, validation and visualization code: `PAR_v2/lart/models/ava_action_prediction.py`
- Examples of how to run experiments: `scripts/launch_action_pred.sh`

## NuScenes Car Trajectory Prediction

This task requires downloading the nuscenes dataset: https://www.nuscenes.org/nuscenes. 

To setup the environment for this task, run the following command. Note that the $HOME path should be updated for the `prefix` in the environment's yaml file.
```
conda env create -f nuscenes_cars.yml
```

This code trains on xy-trajectory data and does not use any environment information. It predicts future trajectories from an input of 2 seconds. 

Experiments can be run with the config `scripts/launch_cars.sh`.

## DexYCB Object Pose Prediction
To setup the environment for this task, run the following command. Note that the $HOME path should be updated for the `prefix` in the environment's yaml file.
```
conda env create -f dexycb_env.yml
```

To get the dataset, add a folder to your home directory (`$HOME`) called `datasets/dexycb` and download the DexYCB dataset in this directory, using [this link](https://dex-ycb.github.io/). 

To train the models presented in the paper for object pose prediction, run the following command, where the 3 arguments specify which GPU index to use, whether you want to do rotation prediction (0,1), and whether you want to run translation prediction (0,1). Note that you can only perform 1 type of prediction task at a time with our current setup.
```
./scripts/launch_dexycb.sh <GPU_NUMBER> <ROT_ONLY_BOOL> <TRANSL_ONLY_BOOL>
```


# Using code for a new predictive problem X
0. Acquire your dataset and extract entities from it.
1. Create a dataloader in `datamodules/components/XX_dataset.py`. Write a main method in that file to test that your dataloader works, write visualization code, etc. 
2. Once the dataloader is setup, create a `datamodules/XX_datamodule.py` file copying the template one, changing import statments and `setup()` method to call your dataset
3. Copy training file `models/template.py` to make `models/XX.py` and modify `step()`` to take in your data, forward it through the model, and compute the loss. 
4. If needed, write a tokenizer and put it in `models/components/XX_tokenizer/tokenizer.py`
4. Copy template yaml config file in `configs/` and replace

```
datamodule:
    _target_: PAR.datamodules.XX_datamodule.XXDataModule
    cfg: ${configs}
    train: ${train}

model:
    _target_: PAR.models.XX.PAR_LitModule
    cfg: ${configs}
```

5. Execute a command as below to run your model

python -m PAR.train -m \
--config-name XX.yaml \
task_name=your_task_name \
trainer=ddp_unused_profiler 


## Citation

If you use the PAR codebase in your research, please use the following BibTeX entry.

```bibtex
@article{thakkar2025polyautoregressive,
  author    = {Thakkar, Neerja and Sadjadpour, Tara, and Rajasegeran, Jathushan, and Ginosar, Shiry, and Malik, Jitendra},
  title     = {Poly-Autoregressive Prediction for Modeling Interactions},
  journal   = {CVPR},
  year      = {2025},
}
```
