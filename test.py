import subprocess
import os
from more_itertools import unique_everseen	
os.system('chmod +x 2nd.sh')
subprocess.call(['./2nd.sh'])
print("give any number")
x=int(input())
print(x)