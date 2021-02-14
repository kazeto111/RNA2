import numpy as np
alphabet = ["A", "C", "G", "T"]
status_size = 4
alphabet_size = 4
INF = 9999999
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

def calc_gamma_psi(sequence_int ,status_transition_matrix_mean, output_p_matrix_mean, init_p_mean):
    sequence_size = len(sequence_int)
    scaling_list = np.zeros(sequence_size, dtype=float)
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

        # スケーリング
        scaling_list[t] = np.sum(A_dp[:, t])
        A_dp[:,t] = A_dp[:,t] / scaling_list[t]

    B_dp = np.zeros((status_size, sequence_size), dtype=float)
    for t in range(sequence_size - 1, -1, -1):
        for i in range(status_size):
            if t == sequence_size - 1:
                B_dp[i][t] = 1

            else:
                sum = 0
                for k in range(status_size):
                    sum += status_transition_matrix_mean[i][k] * output_p_matrix_mean[k][sequence_int[t + 1]] * B_dp[k][t + 1]
                B_dp[i][t] = sum

        # スケーリング
        if t != sequence_size-1:
            B_dp[:,t] = B_dp[:,t] / scaling_list[t]

    gamma_dp = A_dp * B_dp


    psi_dp = np.zeros((status_size, status_size, sequence_size-1), dtype=float)
    for t in range(sequence_size-1):
        for i in range(status_size):
            for j in range(status_size):
                psi_dp[i,j,t] = A_dp[i][t] * status_transition_matrix_mean[i][j] * \
                                output_p_matrix_mean[j][sequence_int[t+1]] * B_dp[j][t+1] / scaling_list[t]

    #print("A_dp")
    #print(A_dp)
    #print("B_dp")
    #print(B_dp)
    #print("gamma_dp")
    #print(gamma_dp)
    #print("psi_dp")
    #print(psi_dp)

    return scaling_list, gamma_dp, psi_dp

def change_parameter(sequence_int ,gamma_dp, psi_dp):
    sequence_size = len(sequence_int)
    status_transition_matrix = np.zeros((status_size, status_size), dtype=float)
    output_p_matrix = np.zeros((status_size, alphabet_size), dtype=float)
    init_p = np.zeros(4, dtype=float)

    for i in range(status_size):
        for j in range(status_size):
            status_transition_matrix[i][j] = np.sum(psi_dp[i][j][0:-1]) / np.sum(gamma_dp[i][0:-1])
        for k in range(alphabet_size):
            sum = 0
            for t in range(sequence_size):
                if sequence_int[t] == k:
                    sum += gamma_dp[i][t]
            output_p_matrix[i][k] = sum / np.sum(gamma_dp[i][:])

        init_p[i] = gamma_dp[0][i]


    return status_transition_matrix, output_p_matrix, init_p

status_transition_matrix_list = np.zeros((number_of_sequence, status_size, status_size), dtype=float)
output_p_matrix_list = np.zeros((number_of_sequence, status_size, alphabet_size), dtype=float)
init_p_list = np.zeros((number_of_sequence, status_size), dtype=float)

iteration = 100
p_list = np.zeros((iteration,number_of_sequence), dtype=float)
for i in range(iteration):
    for j in range(number_of_sequence):
        scaling_list, gamma_dp, psi_dp = calc_gamma_psi(sequence_int_list[j], status_transition_matrix_mean,
                                                        output_p_matrix_mean, init_p_mean)
        status_transition_matrix_list[j], output_p_matrix_list[j], init_p_list[j] \
            = change_parameter(sequence_int_list[j], gamma_dp, psi_dp)

        #対数尤度の格納
        p_list[i,j] = np.sum(np.log(scaling_list))


    status_transition_matrix_mean = np.mean(status_transition_matrix_list, axis=0)
    output_p_matrix_mean = np.mean(output_p_matrix_list, axis=0)
    init_p_mean = np.mean(init_p_list, axis=0)

#ここまでbaumwelch.pyと一緒

#以下viterbiアルゴリズムの関数の定義
def viterbi(sequence_int):
    viterbi_dp = np.zeros((status_size,len(sequence_int)))
    #print(viterbi_dp)
    vkakl = np.zeros((status_size), dtype=float)
    vkakl[0] = -1*INF
    for j in range(len(sequence_int)):
        for i in range(0,status_size):
            if i == 0:
                viterbi_dp[i][j] = -1 * INF
            else:
                if j == 0:
                    viterbi_dp[i][j] = status_transition_matrix_mean[0][i] + output_p_matrix_mean[i-1][sequence_int[j]]
                else:
                    for k in range(1,status_size):
                        vkakl[k] = viterbi_dp[i][j-1] + status_transition_matrix_mean[i][k]
                    viterbi_dp[i][j] = np.amax(vkakl) + output_p_matrix_mean[np.argmax(vkakl)-1][sequence_int[j]]

    return viterbi_dp
#print(viterbi_dp)

#tracebackとファイル出力
path = "/Users/futo/Desktop/output_kadai2-2.txt"
with open(path, "w") as f:
    for i in range(number_of_sequence):
        f.write("sequence")
        f.write(str(i))
        f.write("の隠れ状態")
        f.write("\n")
        viterbi_dp = viterbi(sequence_int_list[i])
        for j in range(len(np.argmax(viterbi_dp, axis=0))):
            f.write(str(np.argmax(viterbi_dp, axis=0)[j]))
        f.write("\n")


