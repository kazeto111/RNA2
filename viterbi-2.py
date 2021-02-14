import numpy as np
INF = 9999999999


path = "/Users/futo/Desktop/test.txt"
with open(path, "r") as f:
    l = f.readlines()
for i in range(len(l)):
    l[i] = l[i].replace("\n","")



#print(l)
alphabet_size = int(l[0])
alphabet = list(l[1].split())
#print(alphabet)
status_size = int(l[2])
status_transition_p = np.zeros((status_size,status_size), dtype=float)
for i in range(status_size):
    for j in range(status_size):
        a = list(map(float, l[3 + i].split()))[j]
        if a == 0:
            status_transition_p[i][j] = -1 * INF
        else:
            status_transition_p[i][j] = np.log(a)
#print(status_transition_p)
output_p = np.zeros((status_size-1,alphabet_size), dtype=float)
for i in range(status_size-1):
    for j in range(alphabet_size):
        a = list(map(float, l[3 + status_size + i].split()))[j]
        if a==0:
            output_p[i][j] = -1 * INF
        else:
            output_p[i][j] = np.log(a)
#print(output_p)

path = "/Users/futo/Desktop/fasta.txt"
with open(path, "r") as f:
    l = f.readlines()[1:]
#print(l)
for i in range(len(l)):
    l[i] = l[i].replace("\n","")
sample_RNA = "".join(l)
#print(sample_RNA)
#print(len(sample_RNA))
#sampleRNAの配列はインデックス番号で保持
sample_RNA_int = np.zeros(len(sample_RNA), dtype=int)
#print(sample_RNA_int)
for i in range(len(sample_RNA)):
    sample_RNA_int[i] = alphabet.index(sample_RNA[i])
#print(sample_RNA_int)
#ファイル入力終わり

#dpの作成
viterbi_dp = np.zeros((status_size,len(sample_RNA_int)))
#print(viterbi_dp)
vkakl = np.zeros((status_size), dtype=float)
vkakl[0] = -1*INF
for j in range(len(sample_RNA_int)):
    for i in range(0,status_size):
        if i == 0:
            viterbi_dp[i][j] = -1 * INF
        else:
            if j == 0:
                viterbi_dp[i][j] = status_transition_p[0][i] + output_p[i-1][sample_RNA_int[j]]
            else:
                for k in range(1,status_size):
                    vkakl[k] = viterbi_dp[i][j-1] + status_transition_p[i][k]
                viterbi_dp[i][j] = np.amax(vkakl) + output_p[np.argmax(vkakl)-1][sample_RNA_int[j]]
#print(viterbi_dp)

#tracebackとファイル出力
path = "/Users/futo/Desktop/output.txt"
with open(path, "w") as f:
    for i in range(len(np.argmax(viterbi_dp, axis=0))):
        f.write(str(np.argmax(viterbi_dp, axis=0)[i]))

