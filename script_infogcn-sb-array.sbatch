#!/bin/bash
#
#SBATCH --job-name=test # Job name for tracking
#SBATCH --partition=sip-ampere  # Partition you wish to use
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8      # Number of CPU threads used by your job
#SBATCH --gres=gpu:1           # Number of GPUs to use 

#SBATCH --array=0  # One job for each of the 4 data cases (modify if needed)
#                    # 0 - cv60 (cross-view)
#                    # 1 - sub60 (cross-subject)
#                    # 2 - set120 (cross-setup)
#                    # 3 - sub120 (cross-subjet)

# 120GB of system RAM will be requested with 2 GPUs specified

#SBATCH --time=2-00:00:00      # Job time limit set to 2 days (48 hours)
#
# ### ## SBATCH --mail-type=END,FAIL,TIME_LIMIT_80 # Events to send email on
#SBATCH --output=joboutput_%A_%a.out # Standard out from your job
#SBATCH --error=joboutput_%A_%a.err  # Standard error from your job

## Initialisation ##
source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

## Execute your program(s) ##
module load CUDA/12.2


############
# Usage
############

# sbatch codes/infogcn/script_infogcn-sb-array.sbatch


############
# InfoGcn model
############


############
# InfoGcn Action Recognition scripts
############

# position encoding model
checkpoint_pos_encode=checkpoint_betaNAPE_ntu-sb.pt
# checkpoint_pos_encode=checkpoint_pos_encode_default.pt
# checkpoint_pos_encode=checkpoint_pos_encode.pt

code=/dcs/pg20/u2034358/codes/infogcn/main.py

# Parameters for training
batch_size=4
epochs=8

# Seed values
seed1=0
seed2=8
seed3=11
seed4=19

# Either default PE type or NAPE
PE_type=NAPE

# Get the data case based on the SLURM array task ID
data_case=(CSet)
current_case=${data_case[$SLURM_ARRAY_TASK_ID]}

# Determine the number of classes based on the data case
if [[ "$current_case" == "CV" || "$current_case" == "CS" ]]; then
  num_class=60
  dataset=ntu
else
  num_class=120
  dataset=ntu120
fi

# Model names
model_name_1=infogcn_${PE_type}_NTU${current_case}${num_class}_seed${seed1}
model_name_2=infogcn_${PE_type}_NTU${current_case}${num_class}_seed${seed2}
model_name_3=infogcn_${PE_type}_NTU${current_case}${num_class}_seed${seed3}
model_name_4=infogcn_${PE_type}_NTU${current_case}${num_class}_seed${seed4}


cd codes/infogcn
conda activate yinka_env

python $code --half=True --batch_size=$batch_size --test_batch_size=$batch_size --save_epoch=5 \
--step 90 100 --num_epoch=$epochs --n_heads=3 --num_worker=4 --k=1 --model_name=$model_name_1 \
--dataset=$dataset --num_class=$num_class --lambda_1=1e-4 --lambda_2=1e-1 --z_prior_gain=3 --seed=$seed1 \
--use_vel=False --datacase=NTU${num_class}_${current_case} --weight_decay=0.0005 --PE_name=$checkpoint_pos_encode \
--num_person=2 --num_point=25 --graph=graph.ntu_rgb_d.Graph --feeder=feeders.feeder_ntu.Feeder &

python $code --half=True --batch_size=$batch_size --test_batch_size=$batch_size --save_epoch=5 \
--step 90 100 --num_epoch=$epochs --n_heads=3 --num_worker=4 --k=1 --model_name=$model_name_2 \
--dataset=$dataset --num_class=$num_class --lambda_1=1e-4 --lambda_2=1e-1 --z_prior_gain=3 --seed=$seed2 \
--use_vel=False --datacase=NTU${num_class}_${current_case} --weight_decay=0.0005 --PE_name=$checkpoint_pos_encode \
--num_person=2 --num_point=25 --graph=graph.ntu_rgb_d.Graph --feeder=feeders.feeder_ntu.Feeder &

wait

python $code --half=True --batch_size=$batch_size --test_batch_size=$batch_size --save_epoch=5 \
--step 90 100 --num_epoch=$epochs --n_heads=3 --num_worker=4 --k=1 --model_name=$model_name_3 \
--dataset=$dataset --num_class=$num_class --lambda_1=1e-4 --lambda_2=1e-1 --z_prior_gain=3 --seed=$seed3 \
--use_vel=False --datacase=NTU${num_class}_${current_case} --weight_decay=0.0005 --PE_name=$checkpoint_pos_encode \
--num_person=2 --num_point=25 --graph=graph.ntu_rgb_d.Graph --feeder=feeders.feeder_ntu.Feeder &

python $code --half=True --batch_size=$batch_size --test_batch_size=$batch_size --save_epoch=5 \
--step 90 100 --num_epoch=$epochs --n_heads=3 --num_worker=4 --k=1 --model_name=$model_name_4 \
--dataset=$dataset --num_class=$num_class --lambda_1=1e-4 --lambda_2=1e-1 --z_prior_gain=3 --seed=$seed4 \
--use_vel=False --datacase=NTU${num_class}_${current_case} --weight_decay=0.0005 --PE_name=$checkpoint_pos_encode \
--num_person=2 --num_point=25 --graph=graph.ntu_rgb_d.Graph --feeder=feeders.feeder_ntu.Feeder &

wait

# compute average of the seeds
MNF=model_names_${current_case}.txt
cat << EOF > $MNF
$model_name_1
$model_name_2
$model_name_3
$model_name_4
EOF

python main.py --datacase=NTU${num_class}_${current_case} --dataset=$dataset --model_names_file=$MNF --avg_best_acc

wait
# Remove created file after it's done
rm $MNF
