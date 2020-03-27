import numpy as np

def une_sbox(numeros_SBOX, L,D):
    """
    numeros_SBOX : 1 a n
    L : Largeur
    D = Depth

    """
    assert L > 0
    assert D > 0
    n = L*D
    assert numeros_SBOX > 0
    assert numeros_SBOX < n+1
    index_start = (numeros_SBOX-1)*8+1
    index_end = 8 + (numeros_SBOX-1)*8
    w_index = 8*L*D+numeros_SBOX
    all_i = [i for i in range(index_start, index_end + 1)]
    all_i_bar = [-1*i for i in range(index_start, index_end + 1)]
    w_index_bar = -w_index
    x_i = all_i[:4]
    y_i = all_i[4:]
    x_i_b = all_i_bar[:4]
    y_i_b = all_i_bar[4:]

    
    c1 =  [x_i_b[0], x_i_b[1],   x_i_b[2],   x_i[3],     y_i_b[2]]
    c2 =  [x_i[0],   x_i_b[1],   x_i_b[2],   x_i_b[3],   y_i_b[1],   y_i_b[2]]
    c3 =  [x_i[0],   x_i_b[1],   x_i[2],     y_i_b[0],   y_i[1],     y_i_b[2]]
    c4 =  [x_i[1],   x_i[2],     y_i_b[0],   y_i_b[1],   y_i_b[2],   y_i[3]  ]
    c5 =  [x_i[1],   x_i_b[2],   x_i_b[3],   y_i[1],     y_i_b[2],   y_i[3]  ]
    c6 =  [x_i[0],   x_i[1],     x_i[2],     y_i_b[0],   y_i_b[1],   y_i[2],     y_i_b[3]]
    c7 =  [x_i[0],   x_i[1],     x_i_b[2],   x_i_b[3],   y_i[1],     y_i[2],     y_i_b[3]]
    c8 =  [y_i_b[0], w_index]
    c9 =  [y_i_b[1], w_index]
    c10 = [y_i_b[2], w_index]
    c11 = [y_i_b[3], w_index]
    c12 = [x_i[0],  x_i_b[1],   x_i_b[2],   x_i[3],     y_i[2]]
    c13 = [x_i[1],  x_i_b[2],   x_i[3],     y_i_b[2],   y_i_b[3]]
    c14 = [x_i_b[1],  x_i[3],   y_i_b[0],   y_i_b[2],   y_i_b[3]]
    c15 = [x_i_b[0], x_i_b[1],  y_i[0],     y_i_b[2],   y_i_b[3]]
    c16 = [x_i[0],  x_i_b[3],   y_i[0],     y_i_b[2],   y_i_b[3]]
    c17 = [x_i_b[0], x_i[2],    x_i[3],     y_i[2],     y_i_b[3]]
    c18 = [x_i_b[0], y_i[0],    y_i_b[1],   y_i[2],     y_i_b[3]]
    c19 = [x_i[0],  x_i_b[1],   x_i[2],     x_i[3],     y_i[3]]
    c20 = [x_i_b[0], x_i[1], x_i[2], x_i[3], y_i[3]]
    c21 = [x_i_b[1], y_i[0], y_i_b[1], y_i_b[2], y_i[3]]
    c22 = [x_i[1], x_i_b[2], x_i[3], y_i[2], y_i[3]]
    c23 = [x_i[1], x_i_b[3], y_i[0], y_i[2], y_i[3]]
    c24 = [x_i[0], y_i[0], y_i_b[1], y_i[2], y_i[3]]
    c25 = [x_i_b[1], y_i[0], y_i[1], y_i[2], y_i[3]]
    c26 = [x_i[0], x_i[1], x_i[2], x_i[3], w_index_bar]
    c27 = [x_i[0], y_i[0], y_i[1], y_i[2], w_index_bar]
    c28 = [x_i[1], y_i[0], y_i[1], y_i[3], w_index_bar]
    c29 = [x_i[0], x_i_b[1], x_i[2], x_i_b[3], y_i[1], y_i_b[3]]
    c30 = [x_i_b[0], x_i[1], x_i_b[3], y_i_b[0], y_i_b[2], y_i_b[3]]
    c31 = [x_i_b[0], x_i[2], y_i_b[0], y_i[1], y_i[2], y_i_b[3]]
    c32 = [x_i_b[0], x_i_b[1], x_i_b[3], y_i_b[0], y_i[2], y_i[3]]
    c33 = [x_i_b[0], x_i[1], x_i_b[2], x_i_b[3], y_i_b[0], y_i_b[1], y_i_b[3]]
    c34 = [x_i_b[1], x_i_b[2], x_i_b[3], y_i_b[0], y_i_b[1], y_i[2], y_i_b[3]]
    c35 = [x_i_b[0], x_i_b[1], x_i[2], x_i_b[3], y_i_b[0], y_i_b[1], y_i[3]]
    c36 = [x_i_b[0], x_i_b[1], x_i_b[2], x_i_b[3], y_i_b[0], y_i[1], y_i[3]]

    return [c1, c2,c3,c4, c5,c6, c7,c8, c9,
            c10, c11, c12,c13,c14, c15,c16, c17,c18, c19,
            c20, c21,c22,c23,c24, c25,c26, c27,c28, c29,
            c30, c31, c32, c33, c34, c35, c36]



#res = une_sbox(1, 4,2)
#print(res)

def compteur(L,D, tau):
    """
    numeros_SBOX : 1 a n
    L : Largeur
    D = Depth

    U = matrice of shape (n − 1, τ )
    U[i,j] = u_i,j

    """
    w_liste  = [8 * L * D + i+1 for i in range( L * D)]
    w_all = np.array(w_liste)
    w_all_bar = -1*np.array(w_liste)
    n = L*D
    U = np.zeros((tau, n-1), dtype=np.int)
    U = U.transpose()
    offset = 9*n
    for i in range(n-1):
        for j in range(tau):
            U[i,j] = offset + 1
            offset = U[i,j]
    U_bar = -1*U
    c_all = []
    c1 = [w_all_bar[0], U[0, 0]]
    c_all.append(c1)
    for j in range(1, tau):
        c_all.append([U_bar[0,j]])
    for i in range(1, n - 1):
        c_all.append([w_all_bar[i], U[i, 0]])
    for i in range(1, n - 1):
        c_all.append([U_bar[i-1, 0], U[i, 0]])
    for i in range(1, n - 1):
        for j in range(1, tau):
            c_all.append([w_all_bar[i], U_bar[i-1, j-1], U[i, j]])
            c_all.append([U_bar[i - 1, j], U[i, j]])
    for i in range(1, n - 1):
        c_all.append([w_all_bar[i], U[i-1, tau-1]])
    c_all.append([w_all_bar[n-1], U[n - 2, tau - 1]])

    return c_all




def lien_entre_couche(L,Li):
    """
    numeros_SBOX : 1 a n
    L : Largeur
    D = Depth

    """
    5+Li*8 ;
    numeros_SBOX_max = int(L)
    X_C_P_1 = np.zeros((4, numeros_SBOX_max), dtype=np.int)
    Y_sortie_C = np.zeros((4, numeros_SBOX_max), dtype=np.int)
    for sb in range(1, numeros_SBOX_max+1):
        index_start = (sb - 1) * 8 + 1
        index_end = 8 + (sb - 1) * 8
        all_i = [i for i in range(index_start, index_end + 1)]
        x_i_C_P_1 = 8*sb+np.array(all_i[:4])
        y_i = 8*(sb-1)+np.array(all_i[4:])
        X_C_P_1[sb-1] = x_i_C_P_1
        Y_sortie_C[sb-1] = y_i

    c = []

    for j in range(L):
        c += [[Y_sortie_C[j][0], X_C_P_1[0][j]]]
        c += [[-Y_sortie_C[j][0], -X_C_P_1[0][j]]]
        c += [[Y_sortie_C[j][1], X_C_P_1[1][j]]]
        c += [[-Y_sortie_C[j][1], -X_C_P_1[1][j]]]
        c += [[Y_sortie_C[j][2], X_C_P_1[2][j]]]
        c += [[-Y_sortie_C[j][2], -X_C_P_1[2][j]]]
        c += [[Y_sortie_C[j][3], X_C_P_1[3][j]]]
        c += [[-Y_sortie_C[j][3], -X_C_P_1[3][j]]]



    return c


def input_output_egalite(x, y, L, D):
    """
    numeros_SBOX : 1 a n
    L : Largeur
    D = Depth

    """

    numeros_SBOX_max = int(L)
    X_C_P_1 = np.zeros((4, numeros_SBOX_max), dtype=np.int)
    Y_sortie_C = np.zeros((4, numeros_SBOX_max), dtype=np.int)
    for sb in range(1, numeros_SBOX_max + 1):
        index_start = (sb - 1) * 8 + 1
        index_end = 8 + (sb - 1) * 8
        all_i = [i for i in range(index_start, index_end + 1)]
        x_i_C_P_1 = 8 * (sb-1) + np.array(all_i[:4])
        y_i = 8 * (sb) + np.array(all_i[4:])
        X_C_P_1[sb - 1] = x_i_C_P_1
        Y_sortie_C[sb - 1] = y_i

    X_C_P_1 = np.concatenate(X_C_P_1, axis=None)
    Y_sortie_C = np.concatenate(Y_sortie_C, axis=None)

    c=[]
    for index_x, xi in enumerate(x):
        if xi:
            c+=[[X_C_P_1[index_x]]]
        else:
            c += [[-X_C_P_1[index_x]]]
    for index_x, xi in enumerate(y):
        if xi:
            c+=[[Y_sortie_C[index_x]]]
        else:
            c += [[-Y_sortie_C[index_x]]]
    return(c)

L = 1
D = 2
n = L*D
tau = 1
res_all = []
for sbox in range(L*D):
    res_all += une_sbox(sbox+1, L,D)
res_all += compteur(L, D, tau)

res_all += lien_entre_couche(L,D)

#res_all +=input_output_egalite([0]*16, [1]*16, L, D)

print(res_all)
print(len(res_all))
max_all = [max(cluase) for cluase in res_all]
print(max(max_all))
print(len(res_all)/max(max_all))

import PyMiniSolvers.minisolvers as minisolvers

solver = minisolvers.MinisatSolver()
for i in range(max(max_all)): solver.new_var(dvar=True)
for iclause in res_all:
    solver.add_clause(iclause)

is_sat = solver.solve()
print(is_sat)
