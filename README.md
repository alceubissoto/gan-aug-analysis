# gan-aug-analysis

## resources - 
   * weights of GANs:
       * soon 
   To use these weights, please refer to each of the original implementations of [PGAN](https://github.com/NVlabs/stylegan2), [StyleGAN2](https://github.com/NVlabs/stylegan2), [pix2pixHD](https://github.com/NVIDIA/pix2pixHD), and [SPADE](https://github.com/NVlabs/SPADE).
   * synthetic images used:
       * https://drive.google.com/drive/folders/10-rLZIuZ1ek2HZmN1jiQdT15x34Px6ea?usp=sharing

## Train splits used:
   #### Anonymization experiments:
   Each different percentage considered is inside the directories, where the directory name represents the percentage used: ```./journal/percentages/[0_0625, 0_125, 0_25, 0_5, 1_0]```
   * Inside each directory, we have a directory for each of the three cases considered:
     * In the root directory, the baseline, without synthetic.
     * ```real_sgan2```, with ratio [1/x : 1/x] (doubling the training set).
     * ```real_14805_sgan2```, with ratio [1/x : x - 1/x] (topping up the training set).
            
   #### Augmentation:
   * Experiments with different GANs:
      * ```./splits/different_gans/[real-all, real-pgan, real-pix2pixhd, real-spade], ./splits/percentages/1_0/real_50p_sgan2>```
   * Experiments with different ratios of synthetics:
      * ```./splits/percentages/1_0/[real_25p_sgan2, real_50p_sgan2, real_100p_sgan2]```
   * Experiments with different sampling methods:
      * ```./splits/percentages/1_0/[real_50p_sgan2 (random), real_50p_sortedsgan2 (best), real_50p_reversesortedsgan2 (worst), real_50p_sgan2diverse (diverse]```
      * ```./splits/percentages/1_0/[real_100p_sgan2 (random), real_100p_sortedsgan2 (best), real_100p_reversesortedsgan2 (worst), real_100p_sgan2diverse (diverse]```
   * Experiments altering the malignant ratio:
      * ```./splits/percentages/1_0/[real_50p_sgan2,  real_50b75m_sgan, real_50b100m_sgan2]```

  ## Test sets csvs:
  * derm7pt (clinical): ```.splits/testsets/derm7pt_clinical.csv```
  * derm7pt (dermato): ```.splits/testsets/derm7pt_dermato.csv```
  * isic19: ```.splits/testsets/isic2019_test.csv```
  * isic20: ```.splits/testsets/isic2020_subset_test.csv```
  * dermofit: ```.splits/testsets/dermofit_test.csv```

## Reproducing our work:
  * We include scripts to execute all experiments: ```run_augmentation.sh``` and ```run_anonymization.sh```
      * Modify it to include the correct path to the images of all the necessary datasets.
      * We used [comet](https://www.comet.ml/site/) and [sacred](https://sacred.readthedocs.io) to organize the huge amount of runs/results. If you don't want to make use of such libraries, modify both ```train_comet_csv.py``` and ```test_comet_csv.py``` files to remove the dependencies. 
      * In the other hand, if you plan to use comet, insert your own API_key and workspace_name in the ```train_comet_csv.py``` file (in the main function), and at the ```run_*.sh``` script.  
      * Check if the images in the splits csv files have the intended path.
