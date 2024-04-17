import numpy as np
from functools import reduce
from itertools import compress
from operator import xor
from scipy.special import erfc, gammaincc

'''Function to convert the list describing the feedback polynomial into a list 
of boolean'''
def poly_to_bool_list(poly):
    bool_list = [1 if coeff in poly else 0 
                 for coeff in range(poly[-1] + 1)]
    return bool_list

'''Function to convert the feedback polynomial from a list of boolean to a list
with the powers of x whose coefficients are different from zero'''
def bool_list_to_poly(bool_list):
    poly = [power for power, coeff in enumerate(bool_list) if coeff]
    return poly

'''Function to convert a 'bytes' variable into a list of boolean'''
def bytes_to_bool_list(bytes_):
    bool_list = [1 if byte & (1 << i) else 0 for byte in bytes_
                 for i in range(7, -1, -1)]
    return bool_list

'''Function to convert a list of boolean into a 'bytes' variable'''
def bool_list_to_bytes(bool_list):
    # Pad the bool_list with False values to ensure its length is a multiple of 8
    remainder = len(bool_list) % 8
    if remainder != 0:
        bool_list = [0] * (8 - remainder) + bool_list

    # Group the bool_list into bytes and convert each byte to an integer
    byte_list = []
    for i in range(0, len(bool_list), 8):
        byte_value = 0
        for j in range(8):
            byte_value |= bool_list[i + j] << (7 - j)
        byte_list.append(byte_value)

    # Convert the byte_list into a bytes object
    return bytes(byte_list)

def right_shift(list, shift = 1):
    return [0] * shift + list[:-shift]


class LFSR():
    def __init__(self, poly, state = None):
        # Feedback polynomial initialization
        self.poly = sorted(poly)
        # LFSR length extraction
        self.length = int(self.poly[-1])
        # LFSR internal state initialization
        if state is None:
            int_all_ones = 2**self.length - 1
            self.state = (int_all_ones).to_bytes(int(np.ceil(self.length/8)),
                                                   byteorder = 'big')
        else:
            self.state = state
        
        self.output = None
        
        self.feedback = None
        
        # Polynomial and state used for calculations
        # print('LFSR Initialization')
        self.bool_poly = poly_to_bool_list(self.poly)[1:]
        # print(f'poly: {self.bool_poly}')
        self.bool_state = bytes_to_bool_list(self.state)[-self.length:]
        # print(f'init state: {self.bool_state}')
    
    '''General function calculating a new iteration of the LFSR. It is used in 
    the __next__, run_teps and cycle functions depending on the required 
    behavior'''
    def new_iteration(self):
        # Set the output to the element '0' of the state list
        self.output = self.bool_state[-1]
        # Calculate the elementwise AND between state and poly lists
        state_poly_and = compress(self.bool_state, self.bool_poly)
        # Calculate the feedback bit as the cumulative XOR of all those results
        self.feedback = reduce(xor, state_poly_and)
        # Calculate the new state firstly shifting it to the right by one,
        # then inserting the feedback bit
        self.bool_state = right_shift(self.bool_state)
        self.bool_state[0] = self.feedback
        # Set the state into 'bytes' format
        self.state = bool_list_to_bytes(self.bool_state)
        
    def __iter__(self):
        return self
    
    def __next__(self):
        self.new_iteration()
        return self.output
    
    def run_steps(self, N=1):
        print(f'{N} iterations of the LFSR have been computed')
        output_list = []
        # Execute the new_iteration function N times
        for _ in range(N):
            self.new_iteration()
            # Append the output to the output list
            output_list.append(self.output)
        return output_list
    
    
    def cycle(self, state=None):
        output_list = []
        starting_state = self.state
        cycle_length = 0
        
        # Since we don't know a priori how long the cycle will be, we use an 
        # infinite loop and we stop it "manually" when the condition is 
        # satisfied
        cycle_completed = False
        while not cycle_completed:
            self.new_iteration()
            cycle_length += 1
            output_list.append(self.output)
            # If the LFSR reaches a state equal to the starting one, the cycle 
            # has been completed
            if self.state == starting_state:
                print(f'LFSR cycle completed. {cycle_length} elements')
                cycle_completed = True
        return output_list
    
    def __str__(self):
        print(f'Feedback polynomial: {self.poly}\nLFSR length: {self.length}\
            \nCurrent state: {self.state}')
        
def berlekamp_massey(b):
    m, r = 0, 1
    p, q = [1], [1]
    for tau in range(len(b)):
        # Discrepancy bit calculation
        d = [p[j] and b[tau-j] for j in range(m+1)]
        d = reduce(xor, d)
        if d:
            # Q*x^r calculation
            qxr = ([0] * r) + q
            # Make sure that p and qxr have the same length padding them
            qxr_padded = qxr + [0] * (len(p) - len(qxr))
            p_padded = p + [0] * (len(qxr) - len(p))
            # Update step
            if 2*m <= tau:
                p, q = [p_ ^ qxr_ for p_, qxr_ in zip(p_padded, qxr_padded)] , p
                m = tau + 1 - m
                r = 0
            else:
                p = [p_ ^ qxr_ for p_, qxr_ in zip(p_padded, qxr_padded)]
        r += 1
    poly = bool_list_to_poly(p)
    return poly

class ShrinkingGenerator():
    def __init__(self,  stateA, stateS, polyA = [0, 2, 5], polyS = [0, 1, 3]):
        self.lfsrA = LFSR(polyA, stateA)
        self.lfsrS = LFSR(polyS, stateS)
        self.output = None
    
    def new_iteration(self):
        # New iteration for lfsrA and lfsrS
        self.lfsrA.new_iteration()
        self.lfsrS.new_iteration()
        # get the output bits of lfsrA and lfsrS
        bita = self.lfsrA.output
        bits = self.lfsrS.output
        # If bits is true, bita can go through, otherwise no output is produced
        if bits:
            self.output = bita
        else:
            self.output = None 

    def __iter__(self):
        return self
    
    def __next__(self):
        self.new_iteration()
        while self.output is None:
            self.new_iteration()
        return self.output
    
    def run_steps(self, N):
        print(f'{N} iterations of the Shrinking Generator have been computed')
        output_list = []
        i = 0
        # Execute the new_iteration until N valid output bits are produced
        while i < N:
            self.new_iteration()
            # Append the output to the output list
            if self.output is not None:
                output_list.append(self.output)
                i += 1
        return output_list
    
def frequency_test(b):
    # Significance level
    alpha = 0.01
    # Compute pi, s and p following the standard
    pi = sum(b) / len(b)
    s = 2 * (len(b) ** 0.5) * np.abs(pi - 0.5)
    p = erfc(s / (2 ** 0.5))
    # Return True (= random sequence) if p>alpha, otherwise return False
    return p > alpha

def block_frequency_test(b, M):
    # Significance level
    alpha  = 0.01
    # Adapt length of b to the block size
    b = b[:len(b) - len(b) % M]
    pi = []
    # Compute number of blocks
    N = int(len(b) / M)
    chi_square = 0
    for i in range(N):
        # Compute pi for each block
        pi.append(sum(b[M*i:M*(i + 1)]) / M)
        # Start computing chi_square
        chi_square += (pi[i] - 0.5)**2
    chi_square *= 4 * M
    # Compute p following the standard
    p = gammaincc(N/2, chi_square/2)
    # Return True (= random sequence) if p>alpha, otherwise return False
    return p > alpha

def runs_test(b):
    # Significance level
    alpha = 0.01
    # Execute frequency test
    fr_res = frequency_test(b)
    pi = sum(b) / len(b)
    if fr_res:
        # If frequency test was passed, proceed with run test
        r_sum = 0
        for i in range(len(b) -1):
            # Compute XNOR on pairs of consecutive bits
            r = not(b[i] ^ b[i+1])
            r_sum += r
        # Compute v and p following the standard
        v = (1 + r_sum)/2/len(b)
        p = erfc(np.abs(v - pi*(1-pi))/(pi*(1 - pi)/2**0.5))
        # Return True (= random sequence) if p>alpha, otherwise return False
        return p > alpha
    else:
        # Frequency test not passed, then return False(= not random sequence)
        return False
