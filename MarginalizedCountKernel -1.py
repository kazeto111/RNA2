import numpy as np
alphabet = ["A", "C", "G", "T"]
status_size = 4
alphabet_size = 4
sequence_int_list = []

fasta_simbol = input()
while True:
    sequence = ""
    while True:
        i = input()
        if len(i) == 0:
            flag = 1
            break
        if i[0] == ">":
            flag = 0
            break
        else:
            sequence += i


    sequence_int = np.zeros((len(sequence)), dtype=int)

    for i in range(len(sequence)):
        sequence_int[i] = alphabet.index(sequence[i])

    sequence_int_list.append(sequence_int)

    if flag == 1:
        break

number_of_sequence = len(sequence_int_list)
#初期化
#初期確率
init_p_mean = np.array([0.7, 0.1, 0.1, 0.1], dtype=float)
status_transition_matrix_mean = np.array([[0.8, 0.1, 0.1, 0.0],
                                     [0.0, 0.8, 0.1, 0.1],
                                     [0.1, 0.0, 0.8, 0.1],
                                     [0.1, 0.0, 0.1, 0.8]], dtype=float)

output_p_matrix_mean = np.array([[0.4, 0.1, 0.1, 0.4],
                            [0.25, 0.25, 0.25, 0.25],
                            [0.1, 0.4, 0.4, 0.1],
                            [0.3, 0.2, 0.2, 0.3]],dtype=float)

#スケーリングなしのfoward_backward_algorithmを別に関数化
def foward_backward_algorithm(sequence_int):
    sequence_size = len(sequence_int)
    A_dp = np.zeros((status_size, sequence_size), dtype=float)
    for t in range(sequence_size):
        for i in range(status_size):
            if t == 0:
                A_dp[i][t] = init_p_mean[i] * output_p_matrix_mean[i][sequence_int[t]]

            else:
                sum = 0
                for k in range(status_size):
                    sum += A_dp[k][t - 1] * status_transition_matrix_mean[k][i]
                A_dp[i][t] = sum * output_p_matrix_mean[i][sequence_int[t]]

    B_dp = np.zeros((status_size, sequence_size), dtype=float)
    for t in range(sequence_size - 1, -1, -1):
        for i in range(status_size):
            if t == sequence_size - 1:
                B_dp[i][t] = 1

            else:
                sum = 0
                for k in range(status_size):
                    sum += status_transition_matrix_mean[i][k] * output_p_matrix_mean[k][sequence_int[t + 1]] * B_dp[k][
                        t + 1]
                B_dp[i][t] = sum

    return A_dp, B_dp

def calc_gamma_MCK(sequence_int, A_dp, B_dp):
    sequence_size = len(sequence_int)
    P = np.sum(A_dp[:, sequence_size-1]) #現在のパラメータを用いた場合の、そのsequenceが得られる確率
    gamma_matrix = np.zeros((alphabet_size, status_size), dtype=float)
    for i in range(alphabet_size):
        for j in range(status_size):
            sum = 0
            for t in range(sequence_size):
                if sequence_int[t] == i:
                    sum += A_dp[j,t] * B_dp[j,t] / P
            gamma_matrix[i,j] = sum
    return gamma_matrix

def calc_MCK(sequence_int_1, sequence_int_2):
    A_dp_1, B_dp_1 = foward_backward_algorithm(sequence_int_1)
    A_dp_2, B_dp_2 = foward_backward_algorithm(sequence_int_2)
    gamma_matrix_1 = calc_gamma_MCK(sequence_int_1, A_dp_1, B_dp_1)
    gamma_matrix_2 = calc_gamma_MCK(sequence_int_2, A_dp_2, B_dp_2)


    return np.sum(gamma_matrix_1 * gamma_matrix_2)

#MCK_matrixはi,j(i<j)にi番目とj番目のsequenceの間のMCKを格納j<=iの時0
MCK_matrix = np.zeros((number_of_sequence, number_of_sequence), dtype=float)

for i in range(number_of_sequence):
    for j in range(i+1, number_of_sequence):
        calc_MCK(sequence_int_list[i], sequence_int_list[j])