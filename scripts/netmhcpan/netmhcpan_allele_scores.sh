#!/bin/bash

peptides_dir=$1

low_filepath=$(find $peptides_dir -type f -name "*low__full.fasta.pep" -print)
intermediate_filepath=$(find $peptides_dir -type f -name "*intermediate__full.fasta.pep" -print)
high_filepath=$(find $peptides_dir -type f -name "*high__full.fasta.pep" -print)

echo "Low immunogenicity sequences filepath: ${low_filepath}"
echo "Intermediate immunogenicity sequences filepath: ${intermediate_filepath}"
echo "High immunogenicity sequences filepath: ${high_filepath}"

# Combine the contents of the files and output to a new file called combined.txt
cat "$low_filepath" "$intermediate_filepath" "$high_filepath" > $peptides_dir/combined.pep

. netmhcpan_allele_scores_one_file.sh $peptides_dir/combined.pep
