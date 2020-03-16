# Copyright 2018 Daniel Selsam. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os
import math
import numpy as np
import random
import argparse
import pickle


# Copyright 2018 Daniel Selsam. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import math



class Problem(object):
    def __init__(self, n_vars, iclauses, is_sat, n_cells_per_batch, all_dimacs):
        self.n_vars = n_vars
        self.n_lits = 2 * n_vars
        self.n_clauses = len(iclauses)

        self.n_cells = sum(n_cells_per_batch)
        self.n_cells_per_batch = n_cells_per_batch

        self.is_sat = is_sat
        self.compute_L_unpack(iclauses)

        # will be a list of None for training problems
        self.dimacs = all_dimacs

    def compute_L_unpack(self, iclauses):
        self.L_unpack_indices = np.zeros([self.n_cells, 2], dtype=np.int)
        cell = 0
        for clause_idx, iclause in enumerate(iclauses):
            vlits = [self.ilit_to_vlit(x, self.n_vars) for x in iclause]
            for vlit in vlits:
                self.L_unpack_indices[cell, :] = [vlit, clause_idx]
                cell += 1

        assert(cell == self.n_cells)

    # TODO(dhs): duplication
    def ilit_to_var_sign(self, x):
        assert(abs(x) > 0)
        var = abs(x) - 1
        sign = x < 0
        return var, sign

    # TODO(dhs): duplication
    def ilit_to_vlit(self, x, n_vars):
        assert(x != 0)
        var, sign = self.ilit_to_var_sign(x)
        if sign: return var + n_vars
        else: return var

class ProblemsLoader(object):
    def __init__(self, filenames):
        self.filenames = filenames

        self.next_file_num = 0
        assert(self.has_next())

    def has_next(self):
        return self.next_file_num < len(self.filenames)

    def get_next(self):
        if not self.has_next():
            self.reset()
        filename = self.filenames[self.next_file_num]
        #print("Loading %s..." % filename)
        with open(filename, 'rb') as f:
            problems = pickle.load(f)
        self.next_file_num += 1
        assert(len(problems) > 0)
        return problems, filename

    def reset(self):
        self.next_file_num = 0



class DataGenerator(object):
    """docstring for DataGenerator."""
    def __init__(self, config, minisolvers):
        super(DataGenerator, self).__init__()
        self.config = config
        self.minisolvers = minisolvers

    def write_dimacs_to(self, n_vars, iclauses, out_filename):
        with open(out_filename, 'w') as f:
            f.write("p cnf %d %d\n" % (n_vars, len(iclauses)))
            for c in iclauses:
                for x in c:
                    f.write("%d " % x)
                f.write("0\n")

    def mk_out_filenames(self, opts, n_vars, t):
        prefix = "%s/sr_n=%.4d_pk2=%.2f_pg=%.2f_t=%d" % \
            (opts.path.dimacs_dir, n_vars, opts.generate_data.p_k_2,
            opts.generate_data.p_geo, t)
        return ("%s_sat=0.dimacs" % prefix, "%s_sat=1.dimacs" % prefix)

    def generate_k_iclause(self, n, k):
        vs = np.random.choice(n, size=min(n, k), replace=False)
        return [v + 1 if random.random() < 0.5 else -(v + 1) for v in vs]

    def gen_iclause_pair(self, opts):
        n = random.randint(opts.generate_data.min_n, opts.generate_data.max_n)
        solver = self.minisolvers.MinisatSolver()
        for i in range(n): solver.new_var(dvar=True)

        iclauses = []

        while True:
            k_base = 1 if random.random() < opts.generate_data.p_k_2 else 2
            k = k_base + np.random.geometric(opts.generate_data.p_geo)
            iclause = self.generate_k_iclause(n, k)

            solver.add_clause(iclause)
            is_sat = solver.solve()
            if is_sat:
                iclauses.append(iclause)
            else:
                break
        iclause_unsat = iclause
        iclause_sat = [- iclause_unsat[0] ] + iclause_unsat[1:]

        return n, iclauses, iclause_unsat, iclause_sat

    def parse_dimacs(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
        i = 0
        while lines[i].strip().split(" ")[0] == "c":
            i += 1
        header = lines[i].strip().split(" ")
        assert(header[0] == "p")
        n_vars = int(header[2])
        iclauses = [[int(s) for s in line.strip().split(" ")[:-1]] for line in lines[i+1:]]
        return n_vars, iclauses

    def mk_dataset_filename(self, opts, n_batches):
        dimacs_path = opts.path.dimacs_dir.split("/")
        dimacs_dir = dimacs_path[-1] if dimacs_path[-1] != "" else dimacs_path[-2]
        return "%s/data_dir=%s_npb=%d_nb=%d.pkl" % (opts.path.out_dir, dimacs_dir,
         opts.generate_data.max_nodes_per_batch, n_batches)


    def solve_sat(self, n_vars, iclauses):
        solver = self.minisolvers.MinisatSolver()
        for i in range(n_vars): solver.new_var(dvar=True)
        for iclause in iclauses: solver.add_clause(iclause)
        is_sat = solver.solve()
        stats = solver.get_stats()
        return is_sat, stats

    def shift_ilit(self, x, offset):
        assert(x != 0)
        if x > 0: return x + offset
        else:     return x - offset

    def shift_iclauses(self, iclauses, offset):
        return [[self.shift_ilit(x, offset) for x in iclause] for iclause in iclauses]

    def mk_batch_problem(self, problems):
        all_iclauses = []
        all_is_sat = []
        all_n_cells = []
        all_dimacs = []
        offset = 0

        prev_n_vars = None
        for dimacs, n_vars, iclauses, is_sat in problems:
            assert(prev_n_vars is None or n_vars == prev_n_vars)
            prev_n_vars = n_vars

            all_iclauses.extend(self.shift_iclauses(iclauses, offset))
            all_is_sat.append(is_sat)
            all_n_cells.append(sum([len(iclause) for iclause in iclauses]))
            all_dimacs.append(dimacs)
            offset += n_vars
        #print(all_iclauses, all_is_sat, all_n_cells, all_dimacs,offset )



        return Problem(offset, all_iclauses, all_is_sat, all_n_cells, all_dimacs)



    def run_main(self):

        for pair in range(self.config.generate_data.n_pairs):
            n_vars, iclauses, iclause_unsat, iclause_sat = self.gen_iclause_pair(self.config)
            out_filenames = self.mk_out_filenames(self.config, n_vars, pair)

            iclauses.append(iclause_unsat)
            self.write_dimacs_to(n_vars, iclauses, out_filenames[0])

            iclauses[-1] = iclause_sat
            self.write_dimacs_to(n_vars, iclauses, out_filenames[1])

        problems = []
        batches = []
        n_nodes_in_batch = 0

        filenames = os.listdir(self.config.path.dimacs_dir)

        # to improve batching
        filenames = sorted(filenames)

        prev_n_vars = None

        for filename in filenames:
            #print(filename)
            n_vars, iclauses = self.parse_dimacs("%s/%s" % (self.config.path.dimacs_dir, filename))
            n_clauses = len(iclauses)
            n_cells = sum([len(iclause) for iclause in iclauses])

            n_nodes = 2 * n_vars + n_clauses
            if n_nodes > self.config.generate_data.max_nodes_per_batch:
                continue

            batch_ready = False
            if (self.config.generate_data.one and len(problems) > 0):
                batch_ready = True
            elif (prev_n_vars and n_vars != prev_n_vars):
                batch_ready = True
            elif (not self.config.generate_data.one) and n_nodes_in_batch + n_nodes > self.config.generate_data.max_nodes_per_batch:
                batch_ready = True

            if batch_ready:
                batches.append(self.mk_batch_problem(problems))
                print("batch %d done (%d vars, %d problems)...\n" % (len(batches), prev_n_vars, len(problems)))
                del problems[:]
                n_nodes_in_batch = 0

            prev_n_vars = n_vars

            is_sat, stats = self.solve_sat(n_vars, iclauses)
            #print(filename, n_vars, iclauses, is_sat)

            problems.append((filename, n_vars, iclauses, is_sat))
            n_nodes_in_batch += n_nodes

        if len(problems) > 0:
            batches.append(self.mk_batch_problem(problems))
            print("batch %d done (%d vars, %d problems)...\n" % (len(batches), n_vars, len(problems)))
            del problems[:]

        # create directory
        if not os.path.exists(self.config.path.out_dir):
            os.mkdir(self.config.path.out_dir)

        dataset_filename = self.mk_dataset_filename(self.config, len(batches))
        print("Writing %d batches to %s...\n" % (len(batches), dataset_filename))
        with open(dataset_filename, 'wb') as f_dump:
            pickle.dump(batches, f_dump)
