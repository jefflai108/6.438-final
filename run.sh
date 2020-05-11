#!/bin/bash

PREFIX="To conclude"
N_SAMPLES=3

for cond in "military" "religion" "politics" "science" "legal" "space" "computers" ; do
  python run_pplm.py -B $cond --cond_text "$PREFIX" --length 100 --gamma 1.5 --num_iterations 3 --num_samples $N_SAMPLES --stepsize 0.03 --window_length 5 --kl_scale 0.01 --gm_scale 0.99 --colorama --sample >> pplm.txt
done

for cond in 2 3 ; do
  python run_pplm.py -D sentiment --class_label $cond --cond_text "$PREFIX" --length 100 --gamma 1.0 --num_iterations 10 --num_samples $N_SAMPLES --stepsize 0.04 --kl_scale 0.01 --gm_scale 0.95 --sample >> pplm.txt
done

# for cond in "negative" "positive" ; do
#   for prefix in "The following is a $cond sentence. The chicken tastes" "The chicken tastes" ; do
#     echo "============================================="
#     echo $cond
#     echo $prefix
#     echo "---------------------------------------------"
#     python generate.py --prefix "$prefix" --condition "$cond" "$@"
#   done
# done
# 
# for cond in "Space" "military" "science" "politics" "computers"; do
#   for prefix in "The issue focused" "The following is an article about $cond. The issue focused" ; do
#     echo "============================================="
#     echo $cond
#     echo $prefix
#     echo "---------------------------------------------"
#     python generate.py --prefix "$prefix" --condition "$cond" "$@"
#   done
# done
# 
# for cond in "positive" "negative" ; do
#   for prefix in "To conclude" "The following is a $cond article about politics. To conclude"; do
#     echo "============================================="
#     echo $cond "politics"
#     echo $prefix
#     echo "---------------------------------------------"
#     python generate.py --prefix "$prefix" --condition "$cond politics" "$@"
#   done
# done
