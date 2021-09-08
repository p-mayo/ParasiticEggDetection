#!/bin/bash
# request resources:
#SBATCH --job-name=ccsc_adni_classification
#SBATCH --nodes=1
#SBATCH --time=14-00:00:00
#SBATCH --mem=64GB
#SBATCH --mail-type=FAIL,END --mail-user=pm15334@bristol.ac.uk
# #SBATCH --array=1,2,3,4,5
## SBATCH --cpus-per-task=28

# module load CUDA/8.0.44-GCC-5.4.0-2.26
# module load libs/cudnn/5.1-cuda-8.0

# on compute node, change directory to 'submission directory':
echo "==============================================================================================================="
echo "|                                     STARTING detect_egg.sh                                             |"
echo "==============================================================================================================="

# cd $PBS_O_WORKDIR
# run your program, timing it for good measure:

#! Record some useful job details in the output file 
echo Running on host `hostname` 
echo Time is `date` 
echo Directory is `pwd` 
echo PBS job ID is $SLURM_JOBID 
echo This jobs runs on the following nodes: 
echo $SLURM_JOB_NODELIST

path=/mnt/storage/home/pm15334/src/ParasiticEggDetection
# source activate cnns 
echo Moving to $path
cd $path

# fold=$SLURM_ARRAY_TASK_ID

# sbatch submit_classify_ccsc.sh /mnt/storage/home/pm15334/src/cauchy_csc/architectures/cauchy.xml 0
# sbatch submit_classify_ccsc.sh /mnt/storage/home/pm15334/src/cauchy_csc/architectures/laplace.xml 0
# sbatch submit_classify_ccsc.sh /mnt/storage/home/pm15334/src/cauchy_csc/architectures/raw.xml 0

prior=$1
seed=$2
imgdim=$3
task=$4
fold=$5
clf_only=$6
now="$(date +'%Y%m%d_%H%M')"

echo "==============================================================================================================="
echo "                                NEW SET OF EXPERIMENTS ($now)"
echo "==============================================================================================================="
echo ""
echo "        XML file  = $xml_dir"
echo "        Seed      = $seed"
echo ""
echo "==============================================================================================================="
echo ""
echo ""

xml_dir=/mnt/storage/home/pm15334/src/cauchy_csc/architectures/tasks/${prior}_csc.xml
clf_dir=/mnt/storage/home/pm15334/src/cauchy_csc/architectures/tasks/cnn_clf.xml
svm_dir=/mnt/storage/home/pm15334/src/cauchy_csc/architectures/tasks/svm_clf.xml

log_dir=/mnt/storage/scratch/pm15334/adni/pytorch/logs/fold_${fold}/
out_dir=/mnt/storage/scratch/pm15334/adni/pytorch/tasks/$task/fold_${fold}/${prior}
train_dir=/mnt/storage/scratch/pm15334/adni/tasks/$task/inner_1/fold_${fold}_train.txt
valid_dir=/mnt/storage/scratch/pm15334/adni/tasks/$task/inner_1/fold_${fold}_test.txt

diagcode=""

if [ "$task" == 'cnvsrest' ]
then
	train_dir=/mnt/storage/scratch/pm15334/adni/tasks/all/inner_1/fold_${fold}_train.txt
	valid_dir=/mnt/storage/scratch/pm15334/adni/tasks/all/inner_1/fold_${fold}_test.txt
	diagcode="-dc cn:0,ad:1,mci:1"
fi
echo $log_dir
echo $out_dir
echo $train_dir
echo $valid_dir


if [ "$prior" != 'raw' ]
then
	if [ "$clf_only" != 'y' ]
	then
		echo "Obtaining the feature maps"
		echo "Runing python run_task.py -xml $xml_dir -s $seed -d $imgdim -od $out_dir -ld $log_dir -td $train_dir -vd $valid_dir"
					 python run_task.py -xml $xml_dir -s $seed -d $imgdim -od $out_dir -ld $log_dir -td $train_dir -vd $valid_dir
	fi
	train_dir=$out_dir/train/l_0_csc/feature_maps/
	valid_dir=$out_dir/valid/l_0_csc/feature_maps/

	echo $train_dir
	echo $valid_dir
fi


echo "===================== CNN ====================="
echo "===================== Varying LEARNING RATE"
echo "Runing python run_task.py -xml $clf_dir -s $seed -d $imgdim -od $out_dir -ld $log_dir -td $train_dir -vd $valid_dir -lr 0.01 $diagcode"
			 python run_task.py -xml $clf_dir -s $seed -d $imgdim -od $out_dir -ld $log_dir -td $train_dir -vd $valid_dir -lr 0.01 $diagcode

echo "Runing python run_task.py -xml $clf_dir -s $seed -d $imgdim -od $out_dir -ld $log_dir -td $train_dir -vd $valid_dir -lr 0.001 $diagcode"
			 python run_task.py -xml $clf_dir -s $seed -d $imgdim -od $out_dir -ld $log_dir -td $train_dir -vd $valid_dir -lr 0.001 $diagcode

echo "Runing python run_task.py -xml $clf_dir -s $seed -d $imgdim -od $out_dir -ld $log_dir -td $train_dir -vd $valid_dir -lr 0.0001 $diagcode"
			 python run_task.py -xml $clf_dir -s $seed -d $imgdim -od $out_dir -ld $log_dir -td $train_dir -vd $valid_dir -lr 0.0001 $diagcode

echo "Runing python run_task.py -xml $clf_dir -s $seed -d $imgdim -od $out_dir -ld $log_dir -td $train_dir -vd $valid_dir -lr 0.00001 $diagcode"
			 python run_task.py -xml $clf_dir -s $seed -d $imgdim -od $out_dir -ld $log_dir -td $train_dir -vd $valid_dir -lr 0.00001 $diagcode
