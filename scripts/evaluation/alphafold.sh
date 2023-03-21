run_alphafold \
--pdb70_database_path=/data/pdb70/pdb70 \
--bfd_database_path /data/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
--uniclust30_database_path /data/uniclust30/uniclust30_2018_08/uniclust30_2018_08 \
--output_dir  $PWD/output/trans7 \
--fasta_paths $PWD/input/trans7.fasta \
--max_template_date=2020-05-14 \
--db_preset=full_dbs \
--use_gpu_relax=True \
--cpus 32