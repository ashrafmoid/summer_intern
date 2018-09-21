import os
import subprocess
scripts=['split_chr.py','break.py','inp.py','run_gen.py']
true_false=['scr1.py','scr2.py','extract30bp.py','slice_intron_positives.py','false_data_coords.py','subtract_script.py','subset_negative_data.py','slice_intron_negatives.py']
for line in scripts:
	os.system('python '+line+'.py')
os.system('chmod +x filter.sh')
subprocess.call(['./filter.sh'])
for line in true_false:
	os.system('python '+line+'.py')

