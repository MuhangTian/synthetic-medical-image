export sweep_id=x4u35ikp
export load_path=data/imgs/train
runs=9
sleep=0.5

for ((i=0; i<$runs; i++))
    do
        sbatch shell/sweep.sh
        sleep $sleep
    done