from tqdm import tqdm

from utils.config import Config
import numpy as np
import PyMiniSolvers.minisolvers as minisolvers



np.random.seed(0)

#-------------------------------------------------------------------------------------------------------------------
# FUNCTION FOR SAT PB

def write_dimacs_to(n_vars, res_all, out_filename):
    with open(out_filename, 'w') as f:
        f.write("p cnf %d %d\n" % (n_vars, len(res_all)))
        for c in res_all:
            for x in c:
                f.write("%d " % x)
            f.write("0\n")

class SAT_pb_generator_miniGIFT(object):
    """docstring for DataGenerator."""

    def __init__(self, L, D, tau, minisolvers, repeat1, reapeat2, add_confition_input_output=True, print_logs = True, check_verif = True):
        super(SAT_pb_generator_miniGIFT, self).__init__()
        self.dico_entree_sortie = {}
        self.print_logs = print_logs
        self.check_verif = check_verif
        self.L = L
        self.D = D
        self.repeat1 = repeat1
        self.repeat2 = reapeat2
        self.tau = tau
        self.minisolvers = minisolvers
        self.nr = D  # Nbre de round
        self.add_confition_input_output = True
        self.change_output_data = False
        self.initialisation = self.get_initiniatiliastio()
        self.S_box = [0x1, 0xa, 0x4, 0xc, 0x6, 0xf, 0x3, 0x9, 0x2, 0xd, 0xb, 0x7, 0x5, 0x0, 0x8, 0xe]
        self.inverseS_box = [0xd, 0x0, 0x8, 0x6, 0x2, 0xc, 0x4, 0xb, 0xe, 0x7, 0x1, 0xa, 0x3, 0x9, 0xf, 0x5]
        if not add_confition_input_output:
            self.res_all += self.initialisation



    def initialisation_sbox(self):
        for sbox in range(L * D):
            self.res_all += self.une_sbox(sbox + 1, L, D)
        self.res_all += self.compteur(L, D, tau)
        if D > 1:
            if L == 4:
                self.res_all += self.rotation_v16(D)


    def get_initiniatiliastio(self):
        nbre_sample_bin = np.random.randint(0, 2 ** self.L, np.random.randint(1, 16, 1))
        initialisation = []
        for nbre_sample_bin_u_index, nbre_sample_bin_u in enumerate(nbre_sample_bin):
            initialisation.append([nbre_sample_bin_u + 1])
        return initialisation


    def une_sbox(self, numeros_SBOX, L, D):
        """
        numeros_SBOX : 1 a n
        L : Largeur
        D = Depth
        """
        assert L > 0
        assert D > 0
        n = L * D
        assert numeros_SBOX > 0
        assert numeros_SBOX < n + 1
        index_start = (numeros_SBOX - 1) * 8 + 1
        index_end = 8 + (numeros_SBOX - 1) * 8
        w_index = 8 * L * (D+1)  + numeros_SBOX - 4
        all_i = [i for i in range(index_start, index_end + 1)]
        all_i_bar = [-1 * i for i in range(index_start, index_end + 1)]
        w_index_bar = -w_index
        x_i = all_i[:4]
        y_i = all_i[4:]
        x_i_b = all_i_bar[:4]
        y_i_b = all_i_bar[4:]
        c1 = [x_i_b[0], x_i_b[1], x_i_b[2], x_i[3], y_i_b[2]]
        c2 = [x_i[0], x_i_b[1], x_i_b[2], x_i_b[3], y_i_b[1], y_i_b[2]]
        c3 = [x_i[0], x_i_b[1], x_i[2], y_i_b[0], y_i[1], y_i_b[2]]
        c4 = [x_i[1], x_i[2], y_i_b[0], y_i_b[1], y_i_b[2], y_i[3]]
        c5 = [x_i[1], x_i_b[2], x_i_b[3], y_i[1], y_i_b[2], y_i[3]]
        c6 = [x_i[0], x_i[1], x_i[2], y_i_b[0], y_i_b[1], y_i[2], y_i_b[3]]
        c7 = [x_i[0], x_i[1], x_i_b[2], x_i_b[3], y_i[1], y_i[2], y_i_b[3]]
        c8 = [y_i_b[0], w_index]
        c9 = [y_i_b[1], w_index]
        c10 = [y_i_b[2], w_index]
        c11 = [y_i_b[3], w_index]
        c12 = [x_i[0], x_i_b[1], x_i_b[2], x_i[3], y_i[2]]
        c13 = [x_i[1], x_i_b[2], x_i[3], y_i_b[2], y_i_b[3]]
        c14 = [x_i_b[1], x_i[3], y_i_b[0], y_i_b[2], y_i_b[3]]
        c15 = [x_i_b[0], x_i_b[1], y_i[0], y_i_b[2], y_i_b[3]]
        c16 = [x_i[0], x_i_b[3], y_i[0], y_i_b[2], y_i_b[3]]
        c17 = [x_i_b[0], x_i[2], x_i[3], y_i[2], y_i_b[3]]
        c18 = [x_i_b[0], y_i[0], y_i_b[1], y_i[2], y_i_b[3]]
        c19 = [x_i[0], x_i_b[1], x_i[2], x_i[3], y_i[3]]
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
        if self.print_logs:
            print()
            print("SBOX NUMERO:", numeros_SBOX)
            print("ENTREES VARIABLES: ", x_i)
            print("SORTIES VARIABLES: ", y_i)
            print("SBOX W index: ", w_index)
            print()

        self.dico_entree_sortie[numeros_SBOX] = {}
        self.dico_entree_sortie[numeros_SBOX]["entrees"] = x_i
        self.dico_entree_sortie[numeros_SBOX]["sorties"] = y_i
        self.dico_entree_sortie[numeros_SBOX]["sbox"] = w_index

        return [c1, c2, c3, c4, c5, c6, c7, c8, c9,
                c10, c11, c12, c13, c14, c15, c16, c17, c18, c19,
                c20, c21, c22, c23, c24, c25, c26, c27, c28, c29,
                c30, c31, c32, c33, c34, c35, c36]


    def compteur(self, L, D, tau):
        """
        numeros_SBOX : 1 a n
        L : Largeur
        D = Depth
        tau = nbre de Sbox inferieure ou egale a tau
        U = matrice of shape (n − 1, τ )
        U[i,j] = u_i,j
        """

        #w_index = 8 * L * (D + 1) + numeros_SBOX - 4
        w_liste = [8 * L * (D+1) + i + 1 - 4  for i in range(L * (D))]
        w_all = np.array(w_liste)
        w_all_bar = -1 * w_all
        n = L * D
        U = np.zeros((tau, n - 1), dtype=np.int)
        U = U.transpose()
        offset = max(w_liste)
        for i in range(n - 1):
            for j in range(tau):
                U[i, j] = offset + 1
                offset = U[i, j]

        self.dico_entree_sortie["sbox"] = w_all

        if self.print_logs:

            print("VARIABLES W POUR COMPTER NBRE DE SBOX: ", w_all)
            print()
            print("VARIABLES ARTIFICIELLLES U POUR COMPTER NBRE DE SBOX: ", U)
            print()

        U_bar = -1 * U
        c_all = []
        c1 = [w_all_bar[0], U[0, 0]]
        c_all.append(c1)
        for j in range(1, tau):
            c_all.append([U_bar[0, j]])
        for i in range(1, n - 1):
            c_all.append([w_all_bar[i], U[i, 0]])
        for i in range(1, n - 1):
            c_all.append([U_bar[i - 1, 0], U[i, 0]])
        for i in range(1, n - 1):
            for j in range(1, tau):
                c_all.append([w_all_bar[i], U_bar[i - 1, j - 1], U[i, j]])
                c_all.append([U_bar[i - 1, j], U[i, j]])
        for i in range(1, n - 1):
            c_all.append([w_all_bar[i], -1*U[i - 1, tau - 1]])
        c_all.append([w_all_bar[n - 1], -1*U[n - 2, tau - 1]])
        ntot = 1 + (tau-1) + (n-2) + (n-2) + (n-2)*(tau-1) + (n-2)*(tau-1)+ (n-2) + 1
        assert len(c_all) == ntot
        return c_all


    def rotation_v16(self, D):
        P_permut = [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15]
        c = []
        for d in range(D):
            numeros_SBOX_all = [1 + 4 * d, 2 + 4 * d, 3 + 4 * d, 4 + 4 * d,
                                5 + 4 * d, 6 + 4 * d, 7 + 4 * d, 8 + 4 * d]
            index_all = []
            index_all2 = []
            for l in range(4):
                numeros_SBOX = numeros_SBOX_all[l]
                index_start = (numeros_SBOX - 1) * 8 + 1 + 4
                index_end = index_start + 4
                index_all += [index_actu for index_actu in range(index_start, index_end)]
            for l in range(4):
                numeros_SBOX = numeros_SBOX_all[l + 4]
                index_start = (numeros_SBOX - 1) * 8 + 1
                index_end = index_start + 4
                index_all2 += [index_actu for index_actu in range(index_start, index_end)]
            if self.print_logs:

                print("VARIABBLES PERMUTATIONS ENTREES: ", index_all)
                print("VARIABBLES PERMUTATIONS SORTIES:", index_all2)
                print()

            if d ==0:
                self.VARIABBLES_PERMUTATIONS_ENTREES_1 = index_all
                self.VARIABBLES_PERMUTATIONS_SORTIES_1 = index_all2
            else:
                self.VARIABBLES_PERMUTATIONS_ENTREES_2 = index_all
                self.VARIABBLES_PERMUTATIONS_SORTIES_2 = index_all2


            for j in range(len(P_permut)):
                c += [[index_all[j], index_all2[P_permut[j]]]]
                c += [[-index_all[j], -index_all2[P_permut[j]]]]
        self.dico_entree_sortie["output"] = index_all2
        self.index_all_exit_new = index_all2
        return c


    def input_output_egalite(self, x, y, L, D, last):
        """
        numeros_SBOX : 1 a n
        L : Largeur
        D = Depth
        """
        numeros_SBOX_all = []
        for l in range(L):
            numeros_SBOX_all.append(l+1)
        for l in range(L):
            numeros_SBOX_all.append(l+1 + (D-1)*L)
        index_all_entry = []
        index_all_exit = []
        for numeros_SBOX in numeros_SBOX_all[:L]:
            index_start = (numeros_SBOX - 1) * 8 + 1
            index_end = 4 + index_start
            index_all_entry += [index_actu for index_actu in range(index_start, index_end)]

        if self.print_logs:
            print("INDEX ENTRES: ", index_all_entry)

        self.INDEX_ENTRES = index_all_entry

        c=[]
        for index_x, xi in enumerate(x):
            if xi:
                c += [[index_all_entry[index_x]]]
            else:
                c += [[-index_all_entry[index_x]]]

        if self.print_logs:
            print("INDEX SORTIES: ", self.index_all_exit_new)
            print()
        for index_y, yi in enumerate(y):
            if yi:
                c+=[[self.index_all_exit_new[index_y]]]
            else:
                c += [[-self.index_all_exit_new[index_y]]]
        return(c)

    def substitution(self, p):
        for i in range(len(p)):
            p[i] = self.S_box[p[i]]
        return p

    def inverseSubstitution(self, p):
        for i in range(len(p)):
            p[i] = self.inverseS_box[p[i]]
        return p

    def diffusion(self, p):
        b = [0, 0, 0, 0]
        b[0] = ((p[0] >> 0) & 0x8) ^ ((p[1] >> 1) & 0x4) ^ ((p[2] >> 2) & 0x2) ^ ((p[3] >> 3) & 0x1);
        b[1] = ((p[0] << 1) & 0x8) ^ ((p[1] >> 0) & 0x4) ^ ((p[2] >> 1) & 0x2) ^ ((p[3] >> 2) & 0x1);
        b[2] = ((p[0] << 2) & 0x8) ^ ((p[1] << 1) & 0x4) ^ ((p[2] >> 0) & 0x2) ^ ((p[3] >> 1) & 0x1);
        b[3] = ((p[0] << 3) & 0x8) ^ ((p[1] << 2) & 0x4) ^ ((p[2] << 1) & 0x2) ^ ((p[3] >> 0) & 0x1);
        for i in range(len(p)):
            p[i] = b[i]
        return p

    def inverseDiffusion(self, p):
        b = [0, 0, 0, 0]
        b[0] = ((p[0] >> 0) & 0x8) ^ ((p[1] >> 1) & 0x4) ^ ((p[2] >> 2) & 0x2) ^ ((p[3] >> 3) & 0x1)
        b[1] = ((p[0] << 1) & 0x8) ^ ((p[1] >> 0) & 0x4) ^ ((p[2] >> 1) & 0x2) ^ ((p[3] >> 2) & 0x1)
        b[2] = ((p[0] << 2) & 0x8) ^ ((p[1] << 1) & 0x4) ^ ((p[2] >> 0) & 0x2) ^ ((p[3] >> 1) & 0x1)
        b[3] = ((p[0] << 3) & 0x8) ^ ((p[1] << 2) & 0x4) ^ ((p[2] << 1) & 0x2) ^ ((p[3] >> 0) & 0x1)
        for i in range(len(p)):
            p[i] = b[i]
        return p

    def add_contants(self, x, c):
        x[0] = x[0] ^ c
        return x

    def addRoundKey(self, c, key):
        for i in range(len(c)):
            c[i] = c[i] ^ key[i]
        return c

    def Encrypt(self, c, key, nr):
        c = self.addRoundKey(c, key)
        for i in range(nr):
            c = self.substitution(c)
            c = self.diffusion(c)
            c = self.add_contants(c, i)
            c = self.addRoundKey(c, key)
        return c

    def Decrypt(self, c, key, nr):
        c = self.addRoundKey(c, key)
        for i in range(nr):
            c = self.add_contants(c, i)
            c = self.inverseDiffusion(c)
            c = self.inverseSubstitution(c)
            c = self.addRoundKey(c, key)
        return c

    def urandom_from_random(self, rng, length):
        if length == 0:
            return b''

        import sys
        chunk_size = 65535
        chunks = []
        while length >= chunk_size:
            chunks.append(rng.getrandbits(
                chunk_size * 4).to_bytes(chunk_size, sys.byteorder))
            length -= chunk_size
        if length:
            chunks.append(rng.getrandbits(
                length * 4).to_bytes(length, sys.byteorder))
        result = b''.join(chunks)
        return result

    def main_step1(self, ct0_init, ct1_init):
        self.res_all = []
        self.initialisation_sbox()
        ct0 = ct0_init
        ct1 = ct1_init
        key = np.random.randint(0, 15, L)
        self.ct0_origin = ct0_init.copy()
        self.ct1_origin = ct1_init.copy()
        self.ct0_final = self.Encrypt(ct0, key, self.nr);
        self.ct1_final = self.Encrypt(ct1, key, self.nr);
        self.ct_bin = []
        for i in range(4):
            self.ct_bin += [int(x) for x in list('{:04b}'.format(self.ct0_origin[i]^self.ct1_origin[i]))]
        self.ct_bin_final = []
        for i in range(4):
            self.ct_bin_final += [int(x) for x in list('{:04b}'.format(self.ct0_final[i]^self.ct1_final[i]))]
        max_all = [max(cluase) for cluase in self.res_all]
        mmin_all = [-min(cluase) for cluase in self.res_all]
        max2_all = max(max_all, mmin_all)
        self.res_all += self.input_output_egalite(self.ct_bin, self.ct_bin_final, L, D, max(max2_all))

    def main_step1_bis(self, ct0_init, ct1_init, ct0_fin, ct1_fin):
        self.res_all = []
        self.initialisation_sbox()
        self.ct0_origin = ct0_init.copy()
        self.ct1_origin = ct1_init.copy()
        self.ct0_final = ct0_fin.copy()
        self.ct1_final = ct1_fin.copy()
        self.ct_bin = []
        for i in range(4):
            self.ct_bin += [int(x) for x in list('{:04b}'.format(self.ct0_origin[i]^self.ct1_origin[i]))]

        self.ct_bin_final = []
        for i in range(4):
            self.ct_bin_final += [int(x) for x in list('{:04b}'.format(self.ct0_final[i]^self.ct1_final[i]))]


        max_all = [max(cluase) for cluase in self.res_all]
        mmin_all = [-min(cluase) for cluase in self.res_all]
        max2_all = max(max_all, mmin_all)
        self.res_all += self.input_output_egalite(self.ct_bin, self.ct_bin_final, L, D, max(max2_all))


    def main_step2(self, ct0_init, ct1_init):
        self.main_step1(ct0_init, ct1_init)
        self.summary_gen_1()
        self.solve_pb()
        if self.is_sat:
            if self.check_verif:
                self.verify()

    def main_step2_bis(self, ct0_init, ct1_init, ct0_fin, ct1_fin):
        self.main_step1_bis(ct0_init, ct1_init, ct0_fin, ct1_fin)
        self.summary_gen_1()
        self.solve_pb()
        if self.is_sat:
            if self.check_verif:
                self.verify()


    def summary_gen_1(self):
        max_all = [max(cluase) for cluase in self.res_all]
        mmin_all = [-min(cluase) for cluase in self.res_all]
        self.max2_all = max(max_all, mmin_all)

        if self.print_logs:

            print("NBRE DE CLAUSES: ", len(self.res_all))
            print("NBRE DE VARIABLES: ", max(self.max2_all))
            print("RATIO : ", len(self.res_all) / max(self.max2_all))
            print("ENTREES - SORTIES: ", self.ct_bin, self.ct_bin_final)

        n_vars = max(self.max2_all)
        self.nbre_clause = len(self.res_all)
        self.nbre_var = max(self.max2_all)
        self.RATIO = len(self.res_all) / max(self.max2_all)




    def solve_pb(self):
        solver = self.minisolvers.MinisatSolver()
        for i in range(max(self.max2_all)): solver.new_var(dvar=True)
        for iclause in self.res_all:
            solver.add_clause(iclause)
        self.is_sat = solver.solve()
        if self.is_sat:
            if not (self.ct0_origin ^ self.ct1_origin == [0, 0, 0, 0]).all():
                if self.print_logs:
                    print()
                    print("IS THE PROBLEM SAT ? ", self.is_sat)
                    print(list(solver.get_model()))
                self.solution = list(solver.get_model())


    def verify(self):
        if self.print_logs:
            print()
            print("-"*100)
            print()
            print("INPUTS SBOX 1-4", [self.solution[v - 1] for v in self.dico_entree_sortie["sbox"][:4]])
            print()
        #print(self.INDEX_ENTRES)
        plot_in = []
        for v in self.INDEX_ENTRES:
            plot_in.append(self.solution[v - 1])
        assert self.ct_bin == plot_in
        if self.print_logs:
            print(plot_in)
            print("RESPECT INPUTS: ", (self.ct_bin == plot_in))
            print()
            print("OUTPUTS SBOX 1-4", [self.solution[v - 1] for v in self.dico_entree_sortie["sbox"][:4]])
            print()
        #print(self.VARIABBLES_PERMUTATIONS_ENTREES_1)
        plot_in = []
        for v in self.VARIABBLES_PERMUTATIONS_ENTREES_1:
            plot_in.append(self.solution[v - 1])
        if self.print_logs:
            print(plot_in)
            print()
            print("INPUTS SBOX 5-8", [self.solution[v - 1] for v in self.dico_entree_sortie["sbox"][4:]])
            print()
        #print(self.VARIABBLES_PERMUTATIONS_SORTIES_1)
        plot_in = []
        for v in self.VARIABBLES_PERMUTATIONS_SORTIES_1:
            plot_in.append(self.solution[v - 1])
        if self.print_logs:

            print(plot_in)
            print()
            print("OUTPUTS SBOX 5-8",
                  [self.solution[v - 1] for v in self.dico_entree_sortie["sbox"][4:]])
            print()
        #print(self.VARIABBLES_PERMUTATIONS_ENTREES_2)
        plot_in = []
        for v in self.VARIABBLES_PERMUTATIONS_ENTREES_2:
            plot_in.append(self.solution[v - 1])
        if self.print_logs:

            print(plot_in)
            print()
            print("INPUTS SBOX 9-12")
            print()
        #print(self.VARIABBLES_PERMUTATIONS_SORTIES_2)
        plot_in = []
        for v in self.VARIABBLES_PERMUTATIONS_SORTIES_2:
            plot_in.append(self.solution[v - 1])
        if self.print_logs:
            print(plot_in)
            print("RESPECT OUTPUTS: ", (self.ct_bin_final == plot_in))
            print()
        assert self.ct_bin_final == plot_in

        self.tau_ici = np.sum([self.solution[v - 1] for v in self.dico_entree_sortie["sbox"]])
        if self.print_logs:

            print("Number SBOX ACTIVE: ", self.tau_ici)
            print("RESPECT W_SBOX: ", (self.tau == np.sum( [self.solution[v - 1] for v in self.dico_entree_sortie["sbox"]])))
            print()
            print()


        assert self.tau >= self.tau_ici

    def main_general_1(self):
        compteur_sat =0
        list_repeat = [i for i in range(self.repeat1)]
        list_repeat2 = [i for i in range(self.repeat2)]

        self.dico_res = {}
        self.dico_res[5] = {}
        self.dico_res[6] = {}
        self.dico_res[7] = {}
        self.dico_res[8] = {}

        for compte, _ in enumerate(tqdm(list_repeat)):
            ct0_init = self.transform_int_to_list(compte)
            ct0_init_origin = ct0_init.copy()
            for compte2, _ in enumerate(list_repeat2):
                ct1_init = self.transform_int_to_list(compte2)
                ct1_init_origin = ct1_init.copy()
                if (ct0_init==ct1_init).all():
                    pass
                else:
                    self.main_step2(ct0_init.copy(), ct1_init.copy())
                    ct0_init = ct0_init_origin.copy()
                    ct1_init = ct1_init_origin.copy()

                    if self.is_sat:
                        self.tau_ici = np.sum([self.solution[v - 1] for v in self.dico_entree_sortie["sbox"]])
                        if self.tau == self.tau_ici:
                            compteur_sat2 = str(self.ct_bin + self.ct_bin_final)
                            if compteur_sat2 in list(self.dico_res[self.tau_ici].keys()):
                                self.dico_res[self.tau_ici][compteur_sat2]["compteur"] += 1
                                self.dico_res[self.tau_ici][compteur_sat2]["inputs_int"].append((ct0_init.copy(), ct1_init))
                            else:
                                compteur_sat += 1
                                self.dico_res[self.tau_ici][compteur_sat2] = {}
                                namesat = "./data/mini_GIFT/" + str(tau) + "_" + str(compteur_sat) + "_dico_res_SAT.DIMACS"
                                write_dimacs_to(self.nbre_var, self.res_all, namesat)
                                """self.dico_res[self.tau_ici][compteur_sat2]["solutions"] = self.solution
                                self.dico_res[self.tau_ici][compteur_sat2]["is_sat"] = self.is_sat
                                self.dico_res[self.tau_ici][compteur_sat2]["nbre_clauses"] = self.nbre_clause
                                self.dico_res[self.tau_ici][compteur_sat2]["nbre_variable"] = self.nbre_var
                                self.dico_res[self.tau_ici][compteur_sat2]["alpha"] = self.nbre_clause / self.nbre_var"""
                                self.dico_res[self.tau_ici][compteur_sat2]["inputs_int"] = [(ct0_init.copy(), ct1_init)]
                                self.dico_res[self.tau_ici][compteur_sat2]["outputs_int"] = [(self.ct0_final.copy(), self.ct1_final.copy())]

                                """self.dico_res[self.tau_ici][compteur_sat2]["delta_inputs_bin"] = self.ct_bin
                                self.dico_res[self.tau_ici][compteur_sat2]["delta_outputs_bin"] = self.ct_bin_final
                                self.dico_res[self.tau_ici][compteur_sat2]["solution_SBOX_active"] = [self.solution[v - 1] for v in self.dico_entree_sortie["sbox"]]
                                self.dico_res[self.tau_ici][compteur_sat2]["solution_entrees1"] = [self.solution[v - 1] for v in self.INDEX_ENTRES]
                                self.dico_res[self.tau_ici][compteur_sat2]["solution_sortie1"] = [self.solution[v - 1] for v in self.VARIABBLES_PERMUTATIONS_ENTREES_1]
                                self.dico_res[self.tau_ici][compteur_sat2]["solution_entrees2"] = [self.solution[v - 1] for v in self.VARIABBLES_PERMUTATIONS_SORTIES_1]
                                self.dico_res[self.tau_ici][compteur_sat2]["solution_sortie2"] = [self.solution[v - 1] for v in self.VARIABBLES_PERMUTATIONS_ENTREES_2]"""
                                #self.dico_res[self.tau_ici][compteur_sat2]["solution_entrees3"] = [self.solution[v - 1] for v in self.VARIABBLES_PERMUTATIONS_SORTIES_2]
                                self.dico_res[self.tau_ici][compteur_sat2]["compteur"] = 1


                                """for k in self.dico_res[self.tau_ici][compteur_sat2]:
                                    if k not in ['clauses', "solutions", "is_sat"]:
                                        print(self.tau_ici, k, self.dico_res[self.tau_ici][compteur_sat2][k])
                                print()"""

        print()
        print("NB DE PB SAT AVEC TAU", self.tau, compteur_sat )
        return self.dico_res

    def transform_int_to_list(self, cmpt):
        bin_compte = '{0:04x}'.format(cmpt)
        int1_ct0 = int(bin_compte[0], 16)
        int2_ct0 = int(bin_compte[1], 16)
        int3_ct0 = int(bin_compte[2], 16)
        int4_ct0 = int(bin_compte[3], 16)
        return np.array([int1_ct0, int2_ct0, int3_ct0, int4_ct0])









#-------------------------------------------------------------------------------------------------------------------
# FUNCTION FOR CIPHER GIFT 16 BITS




#-------------------------------------------------------------------------------------------------------------------
#MAIN



import time

L = 4  # largeur
D = 2  # Profondeur
tau = 5  # Nbre max de Sbox active 5, 6, 7
print_logs = False
check_verif = False
repeat1 = 2 ** 12
repeat2 = 2 ** 12

for tau in [7]:
    sat_gen = SAT_pb_generator_miniGIFT( L, D, tau, minisolvers, repeat1, repeat2, print_logs = print_logs, check_verif = check_verif)
    start = time.time()
    dico_res = sat_gen.main_general_1()
    print()
    print("Time in sec.", time.time() - start)
    print()
    list_key = list(dico_res[tau].keys())






    cpteur = 0

    dico_res_unsat = {}
    dico_res_unsat[5] = {}
    dico_res_unsat[6] = {}
    dico_res_unsat[7] = {}
    dico_res_unsat[8] = {}

    start = time.time()

    for index_key, key in enumerate(tqdm(list_key)):
        ct0_init3 = dico_res[tau][key]["inputs_int"][0][0]
        ct1_init = dico_res[tau][key]["inputs_int"][0][1]
        flag_continue = True

        for index_key2, key2 in enumerate(list_key):
            if key2 !=key:
                if flag_continue:
                    ct0_fin = dico_res[tau][key2]["outputs_int"][0][0]
                    ct1_fin = dico_res[tau][key2]["outputs_int"][0][1]
                    #if ((ct0_fin, ct1_fin) not in dico_res[tau][key]["outputs_int"]):
                    sat_gen.main_step2_bis(ct0_init3.copy(), ct1_init.copy(), ct0_fin.copy(), ct1_fin.copy())
                    if not sat_gen.is_sat:
                        new_key = str(ct0_init3.copy() + ct1_init.copy() + ct0_fin.copy() + ct1_fin.copy())
                        if new_key not in dico_res_unsat[tau].keys():
                            cpteur += 1
                            dico_res_unsat[tau][new_key] = {}
                            namesat = "./data/mini_GIFT/" + str(tau) + "_" + str(cpteur) + "_dico_res_UNSAT.DIMACS"
                            write_dimacs_to(sat_gen.nbre_var, sat_gen.res_all, namesat)
                            if cpteur == index_key+2:
                                flag_continue = False

    print()
    print("Time in sec.", time.time() - start)
    print()
    print("NB DE PB UNSAT AVEC TAU", tau, cpteur )




    del sat_gen, dico_res_unsat, dico_res

