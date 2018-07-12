# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 10:59:05 2018

@author: nsde
"""

import os

######################## Devangari - 14/06
## margin=1,10,100, k=1
#os.system("python dlmnn/main.py -e 200 -k 1 -m 1")
#os.system("python dlmnn/main.py -e 200 -k 1 -m 10")
#os.system("python dlmnn/main.py -e 200 -k 1 -m 100")
## margin=1,10,100, k=3
#os.system("python dlmnn/main.py -e 200 -k 3 -m 1")
#os.system("python dlmnn/main.py -e 200 -k 3 -m 10")
#os.system("python dlmnn/main.py -e 200 -k 3 -m 100")
## margin=1,10,100, k=5
#os.system("python dlmnn/main.py -e 200 -k 5 -m 1")
#os.system("python dlmnn/main.py -e 200 -k 5 -m 10")
#os.system("python dlmnn/main.py -e 200 -k 5 -m 100")
## margin=1000, k=1,3,5
#os.system("python dlmnn/main.py -e 200 -k 1 -m 1000")
#os.system("python dlmnn/main.py -e 200 -k 3 -m 1000")
#os.system("python dlmnn/main.py -e 200 -k 5 -m 1000")

######################## Cifar10 - 15/06
#os.system("python dlmnn/main.py -e 200 -k 3 -m 1")
#os.system("python dlmnn/main.py -e 200 -k 5 -m 1")
#os.system("python dlmnn/main.py -e 200 -k 7 -m 1")
######################## Cifar10 - 15/06
#os.system("python dlmnn/main.py -e 200 -k 5 -m 1")
#os.system("python dlmnn/main.py -e 200 -k 5 -m 10")
#os.system("python dlmnn/main.py -e 200 -k 5 -m 100")
#os.system("python dlmnn/main.py -e 200 -k 5 -m 1000")
#os.system("python dlmnn/main.py -e 200 -k 5 -m 10000")
#os.system("python dlmnn/main.py -e 200 -k 5 -m 0.5") # normalized


######################### Olivetti
os.system("python dlmnn/main.py -e 1000 -k 1 -b 280 -m 100 -n")
