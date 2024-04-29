with open("Dataset.txt",'r') as f:
    datas = f.readlines()

al = {}
for data in datas:
    n,e = data.split()
    n,e = int(n),int(e)
    if n not in al:
        al[n]= []
    al[n].append(e)

keys = al.keys()
rowPointer = [-1]*max(keys)
neigh = []

for i in range(max(keys)):
    if i in al:
        rowPointer[i] = len(neigh)
        for j in al[i]:
            neigh.append(j)

print(neigh)
print(rowPointer)

with open('graph.csr','w') as file:
    file.write(' '.join(map(str,rowPointer))+'\n')
    file.write(' '.join(map(str,neigh))+'\n')

