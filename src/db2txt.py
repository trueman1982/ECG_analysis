import sys
import os

db_dir = "/home/cihang/project/ECG_analysis/database/"
src_dir = "mit-bih-origin"
des_dir = "mit-bih"

for i in xrange(100, 235):
    file_name = os.path.join(db_dir + src_dir, str(i) + ".hea")
    if os.path.isfile(file_name):
        out_file = os.path.join(db_dir + des_dir, str(i) + ".dat")
        os.system("cd %s; /home/cihang/tools/wfdb/bin/rdsamp -r mit-bih-origin/" % (db_dir) + str(i) + " -v -pd > " + out_file)
        out_file = os.path.join(db_dir + des_dir, str(i) + ".ann")
        os.system("cd %s; /home/cihang/tools/wfdb/bin/rdann -r mit-bih-origin/" % (db_dir) + str(i) + " -a atr -v > " + out_file)
