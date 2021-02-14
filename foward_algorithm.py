import numpy as np
alphabet = ["A", "C", "G", "T"]

#foward algorithm
#状態数４(0含めず)
status_size = 4
#初期確率
init_p = np.array([0.7, 0.1, 0.1, 0.1], dtype=float)
status_transition_matrix = np.array([[0.8, 0.1, 0.1, 0.0],
                                     [0.0, 0.8, 0.1, 0.1],
                                     [0.1, 0.0, 0.8, 0.1],
                                     [0.1, 0.0, 0.1, 0.8]], dtype=float)

output_p_matrix = np.array([[0.4, 0.1, 0.1, 0.4],
                            [0.25, 0.25, 0.25, 0.25],
                            [0.1, 0.4, 0.4, 0.1],
                            [0.3, 0.2, 0.2, 0.3]],dtype=float)

sequence = ""
while True:
    i = input()
    if len(i) == 0:
        break
    else:
        sequence += i

sequence_size = len(sequence)
sequence_int = np.zeros((sequence_size), dtype=int)

for i in range(sequence_size):
    sequence_int[i] = alphabet.index(sequence[i])

#Bのスケーリング用のリスト
scaling_list = np.zeros(sequence_size, dtype=float)

A_dp = np.zeros((status_size, sequence_size), dtype=float)
for t in range(sequence_size):
    for i in range(status_size):
        if t == 0:
            A_dp[i][t] = init_p[i] * output_p_matrix[i][sequence_int[t]]

        else:
            sum = 0
            for k in range(status_size):
                sum += A_dp[k][t-1] * status_transition_matrix[k][sequence_int[t]]
            A_dp[i][t] = sum * output_p_matrix[i][sequence_int[t]]

    # スケーリング
    scaling_list[t] = np.sum(A_dp[:, t])
    A_dp[:,t] = np.copy(A_dp[:,t]/np.sum(A_dp[:,t]))

for row in A_dp:
    print(row)

print("scaling_list")
print(scaling_list)





