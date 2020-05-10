#!/bin/bash

for cond in "negative" "positive" ; do
  for prefix in "The following is a $cond sentence. The chicken tastes" "The chicken tastes" ; do
    echo "============================================="
    echo $cond
    echo $prefix
    echo "---------------------------------------------"
    python generate.py --prefix "$prefix" --condition "$cond" "$@"
  done
done

for cond in "Space" "military" "science" "politics" "computers"; do
  for prefix in "The issue focused" "The following is an article about $cond. The issue focused" ; do
    echo "============================================="
    echo $cond
    echo $prefix
    echo "---------------------------------------------"
    python generate.py --prefix "$prefix" --condition "$cond" "$@"
  done
done

for cond in "positive" "negative" ; do
  for prefix in "To conclude" "The following is a $cond article about politics. To conclude"; do
    echo "============================================="
    echo $cond "politics"
    echo $prefix
    echo "---------------------------------------------"
    python generate.py --prefix "$prefix" --condition "$cond politics" "$@"
  done
done
