#This file is for investing if there are multi-line log entries. Normally a single log event occupies one line but not always, e.g. due to stack traces
import loglead.loader as load, loglead.enricher as er, anomaly_detection as ad
import polars as pl
# Processing Hadoop Logs--------------------------------------------------------------------
hadoop_processor = load.HadoopLoader(filename="../../../Datasets/hadoop/",
                                            filename_pattern  ="*.log",
                                     labels_file_name="../../../Datasets/hdfs/anomaly_label.csv")

#In hadoop there are multi-line logentries. 
#The pattern for matching is included allready in the loader
non_match_df, non_match_count, match_df, match_count =  hadoop_processor.lines_not_starting_with_pattern()
#How many lines do not start with expected pattern to start a logline
non_match_count
#How many line match the 
match_count
#Some lines that do not match 
print(non_match_df[0:10,0].to_list())
#Some lines that match
print(match_df[0:10])

# Processing Thunderbird Logs----------------------------------------------------------------------
thunderbird_processor = load.ThunderbirdLoader(filename="../../../Datasets/thunderbird/Thunderbird.log")
thunderbird_processor = load.ThunderbirdLoader(filename="../../../Datasets/thunderbird/Thunderbird_2k.log")
#Thunderbird is labeled on line level. 
#Valid line starts
valid_starts = [
    'R_SERR', 'R_SEG', 'R_SCSI0', 'N_OOM', 'N_CPU', 'N_MAIL', 'R_PAN', 'R_NMI',
    'R_EXT_FS_ABRT1', 'R_EXT_INODE1', 'R_EXT_FS', 'R_PAG', 'N_LUS_LBUG', 'R_CHK_DSK',
    'R_EXT_INODE2', 'N_CALL_TR', 'R_GPF', 'R_ECC', 'N_PBS_SIS', 'N_PBS_BAIL', 'R_MPT',
    'N_PBS_BFD2', 'N_NFS', 'R_EXT_FS_IO', 'N_PBS_CON2', '-', 'N_PBS_BFD1', 'R_VAPI',
    'R_MTT', 'N_PBS_EPI', 'R_SCSI1', 'R_EXT_FS_ABRT2', 'N_AUTH', 'R_RIP'
]

starts_pattern = '|'.join(valid_starts)
pattern = f"^{starts_pattern} \d+ \d{{4}}\.\d{{2}}\.\d{{2}}"
non_match_df, non_match_count, match_df, match_count =  thunderbird_processor.lines_not_starting_with_pattern(pattern = pattern)
#({starts_pattern}) \d+ \d{{4}}\.\d{{2}}\.\d{{2}}"
#As in Hadoop
non_match_count
match_count
print(non_match_df[0:10])
print(match_df[0:10])


# Processing Profilence Logs-------------------------------------------------------------------------------
pro_processor = load.ProLoader(filename="../../../Datasets/profilence/*.txt")
non_match_df, non_match_count, match_df, match_count =  pro_processor.lines_not_starting_with_pattern(pattern = "^\d+ \d{2}\.\d{2}\.\d{4}")
non_match_count
match_count
print(non_match_df[0:10])
print(match_df[0:10])


# Processing HDFS Logs----------------------------------------------------------------
hdfs_processor = load.HDFSLoader(filename="../../../Datasets/hdfs/HDFS.log", 
                                     labels_file_name="../../../Datasets/hdfs/anomaly_label.csv")
non_match_df, non_match_count, match_df, match_count = hdfs_processor.lines_not_starting_with_pattern(pattern = "^\d{6} \d{6}")
non_match_count
match_count
print(non_match_df[0:10])
print(match_df[0:10])