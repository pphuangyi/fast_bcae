gpu_id=$1

fname="results/bcae_3d_result_${gpu_id}.csv"
touch $fname
echo "gpu_id,batch_size,half_mode,frames_per_second" > $fname

for half_mode in none tensor autocast
do
    for bs in 2 4 8 16 32 48 64 80 96 112 128
    do
        cmd="python bcae_3d_inference.py --gpu-id ${gpu_id} --script --half-mode ${half_mode} --batch-size ${bs}"
        output=`$cmd`
        echo "cmd = ${cmd}, result = ${output}"
        echo "${gpu_id},${bs},${half_mode},${output}" >> $fname
    done
done
