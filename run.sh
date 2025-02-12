#!/bin/bash
start=$(date +%s)
python3 ./test_abft.py | tee test_abft.out
end=$(date +%s)
echo "$(($end-$start)) seconds"

