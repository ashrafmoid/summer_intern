import os
import subprocess
#import all the scripts required
cur_dir=os.getcwd()
sys.path.insert(0,cur_dir+'/scripts')
def options(){
	print("Chose from one of the following:")
	print("1 if u want to use our corpus for word2vec.")
	print("2 if u have your own corpus.")
	print("3 if you have true_data and false_data file.")
	print("4 if you have train_true_data,train_false_data,test_true_data and test_false_data")
}

options()
choice=int(raw_input())
if(choice==1):
	temp='GRCh38.p10.genome.fa'
	splt_str='>c'
elif(choice==2):
	print("Enter the filename:"),
	temp=raw_input()
	print("Enter the splitting character that separates one chromosome fromm another:")
	splt_str=raw_input()
elif(choice==3 or choice==4):
	filename=[for i in input("enter the filenames seperated by space:").split()]
#filename has names of file that have different chromosome sequence or name of true and false data file
if(choice==1 or choice ==2):
	seprate_chr_files=splt_chr_code(temp,splt_str)#call to splt_chr.py
	filename=break_into_lines(seprate_chr_files)#call to break.py
#now we will break the corpus files into 3mers

three_mers=break_into_3mers(filename) #call to inp.py
model_name=generate_model()  #call to run_gen.py   
#now generate true and false data for case 1 and 2
if(choice==1 or choice):
	if(choice==1):
		gencode_filename='gencode.v26.annotation.gff3'
	elif(choice==2):
		gencode_filename=raw_input("Enter the full name of gencode file:")
	command="cat "+cur_dir+"/corpus/"+gencode_filename+ "| awk '/\sexon\s|\stranscript\s|\sgene\s/ && /gene_type=protein_coding/' > gencode_protein_coding.gff"
	os.system(command)
	intron_cord=start_and_end_cord('gencode_protein_coding.gff')  #call to scr1.py
	intron_cord=remove_duplicates(intron_cord) #call to scr2.py       
	intron_cord=extract_threshold_bp(intron_cord) #call to extract3bp.py      
	len_pos,positive_seq=get_pos_seq(intron_cord) #call to slice_intron_positives.py        
	#false data generation
	negative_intron=get_false_intron_cords(seprate_chr_files) #call to false_data_coords.py 
	negative_intron=get_actual_false_cords(intron_cord,negative_intron) #call to subtractscript.py 
	actual_negative_seq=get_false_cords_subset(len_pos,negative_intron) #call to subset_negative_data.py
	actual_negative_seq=get_false_intron_seq(actual_negative_seq) #call to slice_intron_negatives.py   <------------------
	#actual_negative_seq and positive_seq are true and false data for case 1 and 2



if(choice==3):



