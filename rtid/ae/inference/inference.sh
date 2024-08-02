gpu_id=$1
model_type=$2

for precision in full half
do
    if [ "${precision}" == "full" ]; then
        half_mode="none"
    elif [ "${precision}" == "half" ]; then
        half_mode="tensor"
    else
        echo "Wrong precision"
        exit 1
    fi

    fname="inference_speed_results/inference_speed_${gpu_id}_${model_type}_${precision}.csv"
    touch $fname
    echo "batch_size,frames_per_second" > $fname

    prefix="python inference.py --script --gpu-id ${gpu_id} --model-type ${model_type} --half-mode ${half_mode}"
    for bs in 1 2 4 8 16 32 48 64 80 96 112 128
    do
        cmd="${prefix} --batch-size ${bs}"
        echo $cmd
        output=`$cmd`
        echo "cmd = ${cmd}, result = ${output}"
        echo "${bs},${output}" >> $fname
    done
done
