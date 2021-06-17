#!/bin/bash

declare -a nets=( "inceptionv4" )

declare -a exp_desc=("Real"  "Real_+_Pix2pixHD"  "Real_+_SPADE"  "Real_+_PGAN" "Real_+_All" "Real_+_25%_SGAN2" "Real_+_50%_SGAN2" "Real_+_100%_SGAN2" "Real_+_50%_Best" "Real_+_50%_Worst" "Real_+_50%_Diverse" "Real_+_100%_Best" "Real_+_100%_Worst" "Real_+_100%_Diverse" "Real_+_50%B75%M_SGAN2" "Real_+_50%B100%M_SGAN2")

declare -a train_csv=("/gan-aug-analysis/splits/percentages/1_0" "/gan-aug-analysis/splits/different_gans/real-pix2pixhd" "/gan-aug-analysis/splits/different_gans/real-spade" "/gan-aug-analysis/splits/different_gans/real-pgan" "/gan-aug-analysis/splits/different_gans/real-all" "/gan-aug-analysis/splits/percentages/1_0/real_25p_sgan2" "/gan-aug-analysis/splits/percentages/1_0/real_50p_sgan2" "/gan-aug-analysis/splits/percentages/1_0/real_100p_sgan2" "/gan-aug-analysis/splits/percentages/1_0/real_50p_sortedsgan2" "/gan-aug-analysis/splits/percentages/1_0/real_50p_reversesortedsgan2" "/gan-aug-analysis/splits/percentages/1_0/real_50p_sgan2diverse""/gan-aug-analysis/splits/percentages/1_0/real_100p_sortedsgan2" "/gan-aug-analysis/splits/percentages/1_0/real_100p_reversesortedsgan2" "/gan-aug-analysis/splits/percentages/1_0/real_100p_sgan2diverse" "/gan-aug-analysis/splits/percentages/1_0/real_50b75m_sgan2" "/gan-aug-analysis/splits/percentages/1_0/real_50b100m_sgan2" )

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
	for id_data in $(seq 0 15); do 
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
