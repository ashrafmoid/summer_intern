import os
import subprocess

def options(){
	print("Chose from one of the following:")
	print("1 if u want to use our corpus for word2vec.")
	print("2 if u have your own corpus.")
	print("3 if you have true_data and false_data file.")
	print("4 if you have train_true_data,train_false_data,test_true_data and test_false_data")
}

options()
cur_dir=os.getcwd()
#sys.path.insert(0,cur_dir+'/codes')
choice=int(input())
filename
if(x==1):
	filename='GRCh38.p10.genome.fa'


elif(x==2):
	print("Enter the filename:"),
	filename=input()

