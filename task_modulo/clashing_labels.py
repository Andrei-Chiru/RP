counter = 0
for i in range(1,10):
    for j in range(1, 10):
        if i % j == i//j:
            print(f"{i} and {j} \n")
            counter+=1
print(counter)