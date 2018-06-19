wget http://www.gems-system.org/datasets/Brain_Tumor1_GEMS.txt
wget http://www.gems-system.org/datasets/Brain_Tumor2_GEMS.txt
wget http://www.gems-system.org/datasets/Prostate_Tumor.txt

python3 main.py Brain_Tumor1_GEMS.txt > log_db_1.txt
python3 main.py 2>/dev/null Brain_Tumor2_GEMS.txt > log_db_2.txt
python3 main.py 2>/dev/null Prostate_Tumor.txt > log_db_3.txt
