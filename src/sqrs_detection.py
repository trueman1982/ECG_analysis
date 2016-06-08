import sys
import os

db_dir = "/home/cihang/project/ECG_analysis/database/"
src_dir = "mit-bih-origin"
des_dir = "mit-bih-r-label"

for i in xrange(100, 235):
    file_name = os.path.join(db_dir + src_dir, str(i) + ".hea")
    if os.path.isfile(file_name):
        os.system("cd %s; /home/cihang/tools/wfdb/bin/sqrs -r mit-bih-origin/" % (db_dir) + str(i))
        out_file = os.path.join(db_dir + des_dir, str(i) + ".rpeak")
        os.system("cd %s; /home/cihang/tools/wfdb/bin/rdann -r mit-bih-origin/" % (db_dir) + str(i) + " -a qrs > " + out_file)
