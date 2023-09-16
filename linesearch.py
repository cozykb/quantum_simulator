import numpy as np
import copy
import matplotlib.pyplot as plt
import pandas as pd
# from scipy.linalg import expm
from tqdm import tqdm
from scipy.special import softmax
# from scipy.stats import entropy


class Simulator():
    def __init__(self, res: int, rows: int, cols: int, q: int, rho: float = 0.1, t: np.ndarray = None, low_voltage: float = -0.1, high_voltage: float = 0.5, nci: bool= False,alpha: int=5, s1_list: list=[0.3,0.3]):
        super().__init__()
        D = rows * cols
        self.n_dots = D
        self.n_q = q
        if type(t) == type(None):
            self.t = 0
        else:
            self.t = t
        self.eta = self.__eta()
        self.res = res
        self.C_dd, self.C_g = self.__calculate_U_mu(rows, cols, rho)
        self.inv_C_dd = np.linalg.inv(self.C_dd)
        self.invC_ddC_g = self.inv_C_dd @ self.C_g
        self.low_voltage = low_voltage
        self.high_voltage = high_voltage
        self.alpha = alpha
        self.s1_list = s1_list
        self.nci = nci

    def __eta(self):  # calculate eta among D dots and q electrons (counting operator)
        state = [[0 for i in range(self.n_dots)]]
        eta = [[0 for i in range(self.n_dots)]]
        for i in range(self.n_q):
            new_state = []
            # temp = []
            for s in state:
                for j in range(self.n_dots):
                    x = copy.deepcopy(s)
                    x[j] += 1
                    new_state.append(x)
            state = []
            [state.append(ns) for ns in new_state if ns not in state]
            eta += state
        return np.array(eta)

    def __calculate_U_mu(self, rows: int, cols: int, rho: float): #from the old simulator
        # from old simulator
        e = 1.602 * 1e-19
        unit = 1.e-18/e
        # Initialize empty tensors for the capacitances
        C_g = np.zeros((rows, cols, rows, cols))
        C_D = np.zeros((rows, cols, rows, cols))

        # For each dot (x-coord)
        for i in range(rows):
            # For each dot (y-coord)
            for j in range(cols):
                # Diagonal capacitance = 1
                C_g[i, j, i, j] = 1

                # Capacitance to dot on the right and left
                if i < rows - 1:
                    C_g[i, j, i+1, j] = rho
                    C_g[i+1, j, i, j] = rho
                # Dot below
                if j < cols - 1:
                    C_g[i, j, i, j+1] = rho
                    C_g[i, j+1, i, j] = rho
                # Diagonal dots
                if i < rows - 1 and j < cols - 1:
                    C_g[i, j, i+1, j+1] = 0.3*rho
                    C_g[i+1, j+1, i, j] = 0.3*rho
                    C_g[i+1, j, i, j+1] = 0.3*rho
                    C_g[i, j+1, i+1, j] = 0.3*rho

                # Same as above
                if i < rows - 1:
                    C_D[i, j, i+1, j] = rho
                    C_D[i+1, j, i, j] = rho
                if j < cols - 1:
                    C_D[i, j, i, j+1] = rho
                    C_D[i, j+1, i, j] = rho
                if i < rows - 1 and j < cols - 1:
                    C_D[i, j, i+1, j+1] = 0.3*rho
                    C_D[i+1, j+1, i, j] = 0.3*rho
                    C_D[i+1, j, i, j+1] = 0.3*rho
                    C_D[i, j+1, i+1, j] = 0.3*rho

        # Reshape into n_dots x n_dots matrix
        C_g = unit * np.reshape(C_g, (self.n_dots, self.n_dots))
        C_D = unit * np.reshape(C_D, (self.n_dots, self.n_dots))

        # (Matrix) multiply the diagonals by random numbers
        C_g = np.diag(np.exp(0.1*np.random.randn(self.n_dots))) @ C_g
        # Add random offsets to all elements
        C_g += unit*0.02 * \
            np.random.rand(self.n_dots**2).reshape(self.n_dots, self.n_dots)

        # Multiply C_D by random numbers too
        gamma = np.exp(0.1*np.random.randn(self.n_dots))
        C_D = np.diag(gamma) @ C_D @ np.diag(gamma)#
        Csum = np.sum(C_g, axis=1)+np.sum(C_D, axis=1)
        C_dd = np.diag(Csum)-C_D
        return C_dd, C_g

    def __form_voltage_space(self, loc: int,  low_voltage: float, high_voltage: float, res:float):# transform the resolution axis into the voltage axis
        voltage_set = np.linspace(low_voltage, high_voltage, res)
        voltage_space = []
        axis = []
        if self.n_dots == 1:
            voltage_space.append(voltage_set[loc])
            axis.append(loc)
            return voltage_space, axis
        while True:
            index = loc % res
            voltage_space.append(voltage_set[index])
            axis.append(index)
            loc //= res
            if loc < res:
                break
        voltage_space.append(voltage_set[loc])
        axis.append(loc)
        return np.array(voltage_space), axis

    def __check_available(self, current_state: np.ndarray):# find states within q-1, q, q+1
        n_q_current = np.sum(current_state)
        available_states = [[], [], []]
        for state in self.eta:
            if np.sum(state) == n_q_current:
                available_states[1].append(state)
            elif np.sum(state) == n_q_current-1:
                available_states[0].append(state)
            elif np.sum(state) == n_q_current+1:
                available_states[2].append(state)
            elif  np.sum(state) == n_q_current+2:
                break
        return available_states
    
    def __inv_S_matrix(self, state):

        def __s(n,alpha,s1):
            n = int(n)
            if n == 0:
                return 1
            if (n & 1) == 0:
                fn = [1/i if i != 0 else 0 for i in range(n, -1, -2)]
            else:
                fn = [1/i if i != 1 else s1 for i in range(n, -1, -2)]
            return n/(alpha*sum(fn))
        # print(state,self.s1_list)
        return np.diag([1/__s(n,self.alpha,s1) for n,s1 in zip(state,self.s1_list)])


    def __e_function(self, voltage: np.ndarray, state: np.ndarray):# the E function
        if self.nci:
            # print(state)
            inv_S = self.__inv_S_matrix(state)
            # print(inv_S)
            # inv_S = np.linalg.inv(S)
            inv_C_dd = inv_S @ self.inv_C_dd @ inv_S * (1/2)
            invC_ddC_g = inv_S @ self.invC_ddC_g
        else:
            inv_C_dd = self.inv_C_dd
            invC_ddC_g = self.invC_ddC_g
        f1 = state.T @ invC_ddC_g @ voltage
        f2 = state.T @ inv_C_dd @ state 
        return f2 - f1


    def __t_function(self, state_one: np.ndarray, state_two: np.ndarray):# the t function for tunneling
        t = 0
        if sum(state_one) == sum(state_two):
            for i in range(self.n_dots):
                for j in range(self.n_dots):
                    if j < i:
                        creation_j = state_two[j] - state_one[j] if (
                            state_two[j] - state_one[j]) == 1 else 0
                        destruction_i = state_one[i] - state_two[i] if (
                            state_one[i] - state_two[i]) == 1 else 0
                        creation_i = state_two[i] - state_one[i] if (
                            state_two[i] - state_one[i]) == 1 else 0
                        destruction_j = state_one[j] - state_two[j] if (
                            state_one[j] - state_two[j]) == 1 else 0
                        t += (self.t[i][j] * creation_j * destruction_i +
                                self.t[j][i] * creation_i*destruction_j)
        return t

    def __epsilon_function(self, voltage: np.ndarray, state_vector_current: np.ndarray, state_vector_next: np.ndarray):# the epsilon function
        epsilon_1 = (self.__e_function(voltage, state_vector_current) +
                     self.__e_function(voltage, state_vector_next)) / 2
        epsilon_2 = np.sqrt((self.__t_function(state_vector_current, state_vector_next))**2 +
                            ((self.__e_function(voltage, state_vector_current) -
                              self.__e_function(voltage, state_vector_next))**2) / 4)
        return epsilon_1 - epsilon_2

    def __state_to_vector(self, state: np.ndarray):
        return np.float64(np.all(self.eta == state, axis=1))

    def __state_to_label(self, state: np.ndarray, eta: np.ndarray = None): # transform state into correspondent label in configuration set
        if type(eta) == type(None):
            eta = self.eta
        return np.argmax(np.float64(np.all(eta == state, axis=1)))
    
    
    def simulate(self, res = None):
        # assert self.nci == False
        if type(res) == type(None):
            res = self.res
        result = np.zeros([res for _ in range(self.n_dots)])
        for index in range(result.size):
            voltage, axis = self.__form_voltage_space(
                index, self.low_voltage, self.high_voltage, res)
            opt_state = self.__check_state(voltage)
            result[tuple(axis)] = self.__state_to_label(opt_state)
        return result
    
    def __check_state(self, voltage):# the Algorithm check state in thesis
        opt_state = np.zeros(self.n_dots)
        for state in self.eta:
            if self.__e_function(voltage, state) < self.__e_function(voltage, opt_state):
                opt_state = state
        available_states = self.__check_available(opt_state)
        # print(opt_state)
        # print(available_states)
        flag = 0
        for electron_config in available_states:
            if len(electron_config) == 0:
                continue
            for state_one in electron_config:
                for state_two in electron_config:
                    if np.any(state_one != state_two):
                        opt_pair = [state_one, state_two]
                        flag = 1
                        break
                if flag != 0:
                    break
            if flag != 0:
                break
        for electron_config in available_states:
            if len(electron_config) == 0:
                continue
            for state_one in electron_config:
                for state_two in electron_config:
                    if np.any(state_one != state_two):
                        if self.__epsilon_function(voltage, state_one, state_two) < self.__epsilon_function(voltage, opt_pair[0], opt_pair[1]):
                            opt_pair = [state_one, state_two]
        # print(opt_pair)
        if self.__epsilon_function(voltage, opt_pair[0], opt_pair[1]) < self.__e_function(voltage, opt_state):
            if self.__e_function(voltage, opt_pair[0]) < self.__e_function(voltage, opt_pair[1]):
                opt_state = opt_pair[0]
            else:
                opt_state = opt_pair[1]
        return opt_state
    
    def __opt_check_state(self, voltage, state):# a little different but the same, perform better in linesearch
        opt_state = state
        available_states = self.__check_available(opt_state)
        for electron_config in available_states:
            if len(electron_config) == 0:
                continue
            for state in electron_config:
                if self.__e_function(voltage, state) < self.__e_function(voltage, opt_state):
                    opt_state = state
        flag = 0
        for electron_config in available_states:
            if len(electron_config) == 0:
                continue
            for state_one in electron_config:
                for state_two in electron_config:
                    if np.any(state_one != state_two):
                        opt_pair = [state_one, state_two]
                        flag = 1
                        break
                if flag != 0:
                    break
            if flag != 0:
                break
        for electron_config in available_states:
            if len(electron_config) == 0:
                continue
            for state_one in electron_config:
                for state_two in electron_config:
                    if np.any(state_one != state_two):
                        if self.__epsilon_function(voltage, state_one, state_two) < self.__epsilon_function(voltage, opt_pair[0], opt_pair[1]):
                            opt_pair = [state_one, state_two]
        if self.__epsilon_function(voltage, opt_pair[0], opt_pair[1]) < self.__e_function(voltage, opt_state):
            if self.__e_function(voltage, opt_pair[0]) < self.__e_function(voltage, opt_pair[1]):
                opt_state = opt_pair[0]
            else:
                opt_state = opt_pair[1]
        return opt_state
    
    def __form_result_dict(self, simulate_result:np.ndarray, res:int, max_q:int):
        output_dict = {}
        for index in range(simulate_result.size):
            voltage, axis = self.__form_voltage_space(
                index, self.low_voltage, self.high_voltage, res)
            label = simulate_result[tuple(axis)]
            if np.sum(self.eta[int(label)]) <= max_q:
                if int(label) not in output_dict.keys():
                    output_dict.update({int(label):[voltage]})
                else:
                    output_dict[int(label)].append(voltage)
        return output_dict            

    def linesearch(self, max_q, num_direction, simulate_result=None, precise=1e-3, safe_barrier = 0.3, check_bound = True, start_step = 0.05, tol = 1e-1):

        if type(simulate_result) == type(None):
            simulate_result = self.simulate(50)
            print("simulation complete")
        else:
            print("import simulation")
        
        output_dataset = {}
        output_startpoint = {}
        simulator_dataset = self.__form_result_dict(simulate_result,50,max_q)
        for n,state_label in enumerate(simulator_dataset.keys()):

            start_voltage = simulator_dataset[state_label][np.random.randint(len(simulator_dataset[state_label]))]
            start_state = self.eta[state_label]
            output_startpoint.update({state_label:start_voltage})
            dir_list = np.array([np.zeros(start_voltage.shape)])
            for _ in range(num_direction):
                inner_step = 0
                outer_step = safe_barrier
                step = start_step
                while True:
                    direct = (1-(-1)) * np.random.random(start_voltage.shape) + (-1)
                    direct /= np.linalg.norm(direct)
                    if direct not in dir_list:
                        break
                np.append(dir_list,[direct], axis=0)
                while True:
                    next_voltage = start_voltage + direct*step

                    if np.any(next_voltage < self.low_voltage) or np.any(next_voltage > self.high_voltage):
                        if check_bound and outer_step - inner_step < tol:
                            break
                    if safe_barrier - step < tol:
                        break
                    if outer_step - inner_step < precise:
                        if state_label not in output_dataset.keys():
                            output_dataset.update({state_label:[[start_voltage + direct*outer_step, start_voltage + direct*inner_step]]})
                        else:
                            output_dataset[state_label].append([start_voltage + direct*outer_step, start_voltage + direct*inner_step])
                        break
                    
                    state = self.__opt_check_state(next_voltage,start_state)
                    if np.all(state == start_state):
                        inner_step = step
                    else:
                        outer_step = step
                    step = (outer_step + inner_step)/2
            print("line search : {:.2%}".format((n+1)/len(simulator_dataset.keys())))
        return output_dataset, output_startpoint
    
    def simple_linesearch(self, start_voltage, direct, precise=1e-2, safe_barrier = 0.5, check_bound = True, start_step = 0.001, tol = 1e-1):
        start_state = self.__check_state(start_voltage)
        outer_step = safe_barrier
        inner_step = 0
        step = start_step
        output_list = []
        start_voltage += np.random.rand(start_voltage.shape[0])*0.005
        while True:
            next_voltage = start_voltage + direct*step

            if np.any(next_voltage < self.low_voltage) or np.any(next_voltage > self.high_voltage):
                if check_bound and outer_step - inner_step < tol:
                    return None
            if safe_barrier - step < tol:
                return None
            if outer_step - inner_step <= precise:
                output_list = [start_voltage + direct*outer_step, start_voltage + direct*inner_step]
                break
            
            state = self.__opt_check_state(next_voltage,start_state)
            if np.all(state == start_state):
                inner_step = step
            else:
                outer_step = step
            step = (outer_step + inner_step)/2
        return output_list
    

    def __new_find_neighbor(self, state, eta):
        n_q = np.sum(state)
        neighbor_set = []
        FULL_SIZE = 2 * len(state) + 1 + len(state) * (len(state) - 1)
        
        def compute_set(n_q):
            output_set = []
            for i in eta:
                if np.sum(i) == n_q:
                    output_set.append(i)
                elif np.sum(i) > n_q:
                    break
            return output_set
        
        def neighbors(state, output_set, distance:int):
            if distance <= 1:
                distance = 1
            for s in output_set:
                if np.all(np.abs(state-s)<=distance):
                    yield s

        i = -1
        flag = True
        while flag:
            if (n_q + i) >= 0:
                q_set = compute_set(n_q + i)
                assert len(q_set) > 0
                for s in neighbors(state, q_set, i):
                    if len(neighbor_set) < FULL_SIZE:
                        neighbor_set.append(s)
                    else:
                        flag = False
                        break
            i += 1
        return np.array(neighbor_set)


    def __form_hamiltonian(self, voltage:np.ndarray, state_set: np.ndarray):
        H = np.zeros((len(state_set),len(state_set)))
        for i in range(len(state_set)):
            for j in range(len(state_set)):
                if i == j:
                    H[i][j]=self.__e_function(voltage, state_set[i])
                else:
                    H[i][j]=self.__t_function(state_set[i], state_set[j])
        return H

    def __form_A(self, state_set:np.ndarray, pick_dot:int):
        return np.diag(state_set[:,pick_dot])

    def __form_E_label(self, opt_state:np.ndarray, E_state:np.ndarray):
        # label = np.argmax(np.float64(np.all(eta == opt_state, axis=1)))
        # l = np.sum(E_state-opt_state)
        # l = np.sum(E_state) % int(np.sum(E_state))
        l = np.mean(np.square(E_state-opt_state))
        # l = np.sum(E_state)
        return l
    
    def __normalized_expm(self, M):
        D, U = np.linalg.eig(M)
        D = softmax(D)
        return U @ np.diag(D) @ U.T

    def entropy(self, p):
        return np.sum(-p*np.log10(p))

    def sensor_simulator_core(self, pick_sensors:list[int], voltage:list[float], beta:float, eta, FULL_OBS):
        if FULL_OBS:
            N_state_set = self.eta
            opt_state_all = self.eta[0]
            opt_state_dots = opt_state_all[pick_sensors]
        else:
            opt_state_all = self.__check_state(voltage)
            opt_state_dots = opt_state_all[pick_sensors]
            N_state_set = self.__new_find_neighbor(opt_state_all, eta)
        # print(N_state_set)
        # assert 0
        H = self.__form_hamiltonian(voltage, N_state_set)
        E_state = []
        for n_dot in pick_sensors:
            A = self.__form_A(N_state_set, n_dot)
            expmH = self.__normalized_expm(-beta*H)
            rho = (expmH/np.trace(expmH))
            # print(self.__state_to_label(opt_state_all))
            # print(p[self.__state_to_label(opt_state_all)])
            E_state_i = np.trace(A @ rho)
            E_state.append(E_state_i)
        E_state = np.array(E_state)
        return E_state, opt_state_all, opt_state_dots
    
    def sensor_simulator_2dots(self, pick_sensor:list[int], beta:float, res:int = None, FULL_OBS:bool = False):
                # assert self.nci == False
        def _dots_eta():
            state = [[0 for i in range(self.n_dots)]]
            eta = [[0 for i in range(self.n_dots)]]
            for i in range(self.n_q+3):
                new_state = []
                for s in state:
                    for j in range(self.n_dots):
                        x = copy.deepcopy(s)
                        x[j] += 1
                        new_state.append(x)
                state = []
                [state.append(ns) for ns in new_state if ns not in state]
                eta += state
            return np.array(eta)
        eta = _dots_eta()
        if FULL_OBS:
            print("size of eta :",len(self.eta))
        if type(res) == type(None):
            res = self.res
        l_E_result = np.zeros([res for _ in range(self.n_dots)])
        E_result = []
        for index in tqdm(range(l_E_result.size)):
            voltage, axis = self.__form_voltage_space(
                index, self.low_voltage, self.high_voltage, res)
            E_state, opt_state_all, opt_state_dots = self.sensor_simulator_core(pick_sensor, voltage, beta, eta,FULL_OBS)
            E_result.append(E_state)
            l_E_result[tuple(axis)] = self.__form_E_label(opt_state_dots, E_state)
        return l_E_result, E_result

    def sensor_simulator_3dots(self, pick_dots:list[int], pick_sensors:list[int], v_sensors:list[float], beta:float, res:int = None, FULL_OBS:bool = False):
                # assert self.nci == False
        def _dots_eta():
            state = [[0 for i in range(self.n_dots)]]
            eta = [[0 for i in range(self.n_dots)]]
            for i in range(self.n_q+3):
                new_state = []
                for s in state:
                    for j in range(self.n_dots):
                        x = copy.deepcopy(s)
                        x[j] += 1
                        new_state.append(x)
                state = []
                [state.append(ns) for ns in new_state if ns not in state]
                eta += state
            return np.array(eta)
        eta = _dots_eta()
        if FULL_OBS:
            print("size of eta :",len(self.eta))
        if type(res) == type(None):
            res = self.res
        l_E_result = np.zeros([res for _ in range(len(pick_dots))])
        E_result = []
        for index in tqdm(range(l_E_result.size)):
            v_dots, axis = self.__form_voltage_space(
                index, self.low_voltage, self.high_voltage, res)
            voltage = np.zeros(len(pick_dots)+len(pick_sensors))
            voltage[pick_dots] = v_dots
            voltage[pick_sensors] = v_sensors
            # print(voltage)
            E_state, opt_state_all, opt_state_dots = self.sensor_simulator_core(pick_sensors, voltage, beta, eta,FULL_OBS)
            E_result.append(E_state)
            l_E_result[tuple(axis)] = self.__form_E_label(opt_state_dots, E_state)
        return l_E_result, E_result