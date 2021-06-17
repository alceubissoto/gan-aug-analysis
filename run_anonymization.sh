#!/bin/bash

declare -a nets=( "inceptionv4" )

declare -a exp_desc=("6.25%Real" "6.25%Real_+_SGAN2" "6.25%Real_+_FillSGAN2" "12.5%Real" "12.5%Real_+_SGAN2" "12.5%Real_+_FillSGAN2" "25%Real" "25%Real_+_SGAN2" "25%Real_+_FillSGAN2" "50%Real" "50%Real_+_SGAN2")

declare -a train_csv=("/gan-aug-analysis/splits/percentages/0_0625" "/gan-aug-analysis/splits/percentages/0_0625/real_sgan2" "/gan-aug-analysis/splits/percentages/0_0625/real_14805_sgan2" "/gan-aug-analysis/splits/percentages/0_125" "/gan-aug-analysis/splits/percentages/0_125/real_sgan2" "/gan-aug-analysis/splits/percentages/0_125/real_14805_sgan2" "/gan-aug-analysis/splits/percentages/0_25" "/gan-aug-analysis/splits/percentages/0_25/real_sgan2" "/gan-aug-analysis/splits/percentages/0_25/real_14805_sgan2" "/gan-aug-analysis/splits/percentages/0_5" "/gan-aug-analysis/splits/percentages/0_5/real_sgan2")

isic19_root=""
isic20_root=""
derm7pt_root=""
edinburgh_root=""

val_csv="/gan-aug-analysis/splits/isic2019-val.csv"

export COMET_API_KEY=""

SPLITS=9
GPU=$1
MODEL_LIST=nets[@]


for split in $(seq 0 $SPLITS); do
	for id_data in $(seq 0 10); do 
		for net in "${nets[@]}"; do
			unset COMET_EXPERIMENT_KEY
			CUDA_VISIBLE_DEVICES=$GPU python3 train_comet_csv.py with \
				train_root=${isic19_root} train_csv=${train_csv[id_data]}/train_${split}.csv epochs=100\
				val_root=${isic19_root} val_csv=${val_csv} model_name="$net" exp_desc=${exp_desc[id_data]} exp_name="gans.train_${exp_desc[id_data]##*/}.${net}.split${split}"
			comet_key=$(cat gans.train_${exp_desc[id_data]##*/}.${net}.split${split}.txt)
                        export COMET_EXPERIMENT_KEY=${comet_key}
			rm gans.train_${exp_desc[id_data]##*/}.${net}.split${split}.txt
			printenv | grep COMET_EXPERIMENT_KEY
                        CUDA_VISIBLE_DEVICES=$GPU python3 test_comet_csv.py results-comet-gans/gans.train_${exp_desc[id_data]##*/}.${net}.split${split}/checkpoints/model_best.pth ${isic19_root} /gan-aug-analysis/testsets/isic2019-test.csv -n 50 -p -jpg -name isic2019 > results-comet-gans/gans.train_${exp_desc[id_data]##*/}.${net}.split${split}/test_isic2019.txt
                        CUDA_VISIBLE_DEVICES=$GPU python3 test_comet_csv.py results-comet-gans/gans.train_${exp_desc[id_data]##*/}.${net}.split${split}/checkpoints/model_best.pth ${isic20_root} /gan-aug-analysis/testsets/isic2020-subset-test.csv -n 50 -p -jpg -name isic2020 > results-comet-gans/gans.train_${exp_desc[id_data]##*/}.${net}.split${split}/test_isic2020.txt
			CUDA_VISIBLE_DEVICES=$GPU python3 test_comet_csv.py results-comet-gans/gans.train_${exp_desc[id_data]##*/}.${net}.split${split}/checkpoints/model_best.pth ${derm7pt_root} /gan-aug-analysis/testsets/derm7pt_dermato.csv  -n 50 -p -name atlas-dermato > results-comet-gans/gans.train_${exp_desc[id_data]##*/}.${net}.split${split}/test_derm7pt_dermato.txt
			CUDA_VISIBLE_DEVICES=$GPU python3 test_comet_csv.py results-comet-gans/gans.train_${exp_desc[id_data]##*/}.${net}.split${split}/checkpoints/model_best.pth ${derm7pt_root} /gan-aug-analysis/testsets/derm7pt_clinical.csv  -n 50 -p -name atlas-clinical > results-comet-gans/gans.train_${exp_desc[id_data]##*/}.${net}.split${split}/test_derm7pt_clinical.txt
			CUDA_VISIBLE_DEVICES=$GPU python3 test_comet_csv.py results-comet-gans/gans.train_${exp_desc[id_data]##*/}.${net}.split${split}/checkpoints/model_best.pth ${edinburgh_root} /gan-aug-analysis/testsets/dermofit.csv  -n 50 -p  -name edinburgh-all> results-comet-gans/gans.train_${exp_desc[id_data]##*/}.${net}.split${split}/test_dermofit.txt
		done
	done
done
