import os

os.system("conda activate pytorch")
# os.system("python -u test.py > testout1990.txt")
# os.system("python -u test.py --epoch 1500 > testout1500.txt")
os.system("python -u test.py --epoch 650 --net_name jingjin_LF_10-6 --weight_smooth 0.000001")
os.system("python -u test.py --epoch 760 --net_name jingjin_LF_10-5 --weight_smooth 0.00001")
print("\nfinish\n")
