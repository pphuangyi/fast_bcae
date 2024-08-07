gpu_id=$1
machine=$2
model_type=$3

folder="inference_speed_results/inference_speed_${machine}"

if [ -d "$folder" ]; then
    echo "folder $folder already exists."
else
    mkdir -p "$folder"
fi


for precision in full half
do
    if [ "${precision}" == "full" ]; then
        half_mode="none"
        comp="compiled"
    elif [ "${precision}" == "half" ]; then
        half_mode="tensor"
        comp="scripted"
    else
        echo "Wrong precision"
        exit 1
    fi

    fname="${folder}/inference_speed_${gpu_id}_${model_type}_${precision}.csv"
    if [ -f "$fname" ]; then
        echo "Append results to ${fname}"
    else
        touch $fname
        echo "Save results to ${fname}"
        echo "batch_size,frames_per_second" > $fname
    fi

    prefix="python inference.py --${comp} --gpu-id ${gpu_id} --model-type ${model_type} --half-mode ${half_mode}"
    for bs in 1 2 4 8 16 32 48 64 80 96 112 128 144 196 256
    do
        cmd="${prefix} --batch-size ${bs}"
        echo $cmd
        output=`$cmd`
        echo "cmd = ${cmd}, result = ${output}"
        echo "${bs},${output}" >> $fname
    done
done
