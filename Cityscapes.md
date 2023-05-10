### Cityscapes
Download from here: [Cityscapes dataset](https://www.cityscapes-dataset.com/).

## Training

For training on Cityscapes, copy the dataset to `${TMPDIR}/cityscapes/`, and, in `params.yml`, set:
```
dataset_file: datasets.cityscapes
```
then run:
```
python ddpm_train.py params_train_cdm_dino.yml
``` 

to use multiple gpus, inside params_train_cdm_dino.yml set: 
```
distributed: yes
```
note: this defaults to using all visible gpus as we assume a slurm-managed cluster
but reducing that to specific devices should be straightforward inside ddpm_train.py by using `os.environ["CUDA_VISIBLE_DEVICES"] = 0,1`

## Evaluation with cdm_dino_256x512 checkpoint

To run the evaluation download and extract the checkpoint directory and place inside `checkpoints/` :
(The directory should be `checkpoints/cdm_dino_256x512/cdm_dino_256x512.pt`)

```
python ddpm_eval.py params_eval_cdm_dino_256x512.yml
``` 

note: predictions and labels are saved by the above script and then the official cityscapes script is used to get the metrics

