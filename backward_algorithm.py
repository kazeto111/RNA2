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

B_dp = np.zeros((status_size, sequence_size), dtype=float)
for t in range(sequence_size-1, -1, -1):
    for i in range(status_size):
        if t == sequence_size-1:
            B_dp[i][t] = 1

        else:
            sum = 0
            for k in range(status_size):
                sum += status_transition_matrix[i][k] *  output_p_matrix[k][sequence_int[t+1]] * B_dp[k][t+1]
            B_dp[i][t] = sum

    # スケーリング

for row in B_dp:
    print(row)