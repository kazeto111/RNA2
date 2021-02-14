import numpy as np

alphabet_size = int(input())
alphabet = list(input().split())
status_size = int(input())
status_transition_p = []
for i in range(status_size):
    status_transition_p.append(list(map(float,input().split())))
    print("status_transition_p",status_transition_p)
output_p = []
for i in range(alphabet_size):
    output_p.append(list(map(float,input().split())))
    print("output_p",output_p)

sample_RNA = input()
print("sample_RNA",sample_RNA)
sample_RNA_size = len(sample_RNA)
print(sample_RNA_size)

#sampleRNAのアルファベットをリストalphabetのインデックスに置き換えたリストsample_RNA_intを作成する
sample_RNA_int = [0] * sample_RNA_size
for i in range(sample_RNA_size):
    #try:
    sample_RNA_int[i] = alphabet.index(sample_RNA_int[i])
    #except ValueError:
        #print("error")
        #break


viterbi_list = [0] * status_size #ある時刻tでの各状態のviterbi変数を保存したリスト(順次更新)　#
status_transition_list = [0] * sample_RNA_size # 最大確率の隠れ状態の遷移を保存したリスト
vkakl = np.zeros((status_size,status_size)) # 時刻tにおけるviterbi変数と遷移確率の積を全て格納する行列(順次更新)
for j in range(sample_RNA_size):
    for i in range(1, status_size):
        if j == 0:
            print("i,j",i,j)
            print(sample_RNA_int[j])
            viterbi_list[i] = status_transition_p[0][i] * output_p[i-1][sample_RNA_int[j]]
        else:
            viterbi_list[i] = np.max(vkakl, axis=0)[i] * output_p[np.argmax(vkakl, axis=0)[i]][i]
    for k in range(1,status_size):
        for l in range(1,status_size):
            vkakl[k,l] = viterbi_list[k] * status_transition_p[k][l]

    status_transition_list[j] = np.unravel_index(np.argmax(vkakl), vkakl.shape)[0]
    print(j)


print(status_transition_list)