import itertools

for idx, val in enumerate(itertools.cycle(range(4))):
    print(val)
    print(idx)
    if idx>20:
        break