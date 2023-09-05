import os

print('results are created in the results folder')

os.makedirs('results', exist_ok=True)

f = open("results/results.txt", "w+")
f.write("Results\n")

print('Directory created')