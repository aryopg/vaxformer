#!/bin/bash

peptides_dir=$1

low_filepath=$(find $peptides_dir -type f -name "*low__full.fasta.pep" -print)
intermediate_filepath=$(find $peptides_dir -type f -name "*intermediate__full.fasta.pep" -print)
high_filepath=$(find $peptides_dir -type f -name "*high__full.fasta.pep" -print)

echo ${low_filepath}
echo ${intermediate_filepath}
echo ${high_filepath}

# Combine the contents of the files and output to a new file called combined.txt
cat "$low_filepath" "$intermediate_filepath" "$high_filepath" > $peptides_dir/combined.pep

for mhc in A01:01 A02:01 A03:01 A24:02 A26:01 B07:02 B08:01 B27:05 B39:01 B40:01 B58:01 B15:01
do
    netMHCpan -p $peptides_dir/combined.pep -a HLA-${mhc} > $peptides_dir/combined_${mhc:0:3}${mhc:4:2}.out
done