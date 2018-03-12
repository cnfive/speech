# coding:UTF-8

#from tqdm import trange
from time import sleep
import time
#from tqdm import *
import pickle
dic=[]
filepath="/home/yang/speech/doc/trans/train.word.txt"
file = open(filepath) 
filename=[]
file_and_id={}
max_len=0
for line in file:
    #pass # do something
    #print(line)


    s_t=line.strip().split(" ")
    length=len(s_t)
    if length>max_len:
       max_len=length
print("max_len:",max_len)


file.close()



