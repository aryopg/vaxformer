#!/bin/bash

filepath=$1
for mhc in A01:01 A02:01 A03:01 A24:02 A26:01 B07:02 B08:01 B27:05 B39:01 B40:01 B58:01 B15:01
do
    netMHCpan -p ${filepath} -a HLA-${mhc} > ${filepath}_${mhc:0:3}${mhc:4:2}.out
done