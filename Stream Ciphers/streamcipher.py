import numpy as np
from functools import reduce
from itertools import compress
from operator import xor

# def string_to_bytes_state(string_state):
#     int_state = int(string_state, 2)
#     bytes_state = int_state.to_bytes(
#         int(np.ceil(int_state.bit_length()/ 8)), 'big')
#     return bytes_state

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
        print('LFSR Initialization')
        self.bool_poly = poly_to_bool_list(self.poly)[1:]
        print(f'poly: {self.bool_poly}')
        self.bool_state = bytes_to_bool_list(self.state)[-self.length:]
        print(f'init state: {self.bool_state}')
    
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
        print(f'\nHere are {N} iterations of the LFSR')
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
        while True:
            self.new_iteration()
            cycle_length += 1
            output_list.append(self.output)
            # If the LFSR reaches a state equal to the starting one, the cycle 
            # has been completed
            if self.state == starting_state:
                print(f'\nCycle completed. {cycle_length} elements')
                break
        return output_list
    
    def __str__(self):
        print(f'\nFeedback polynomial: {self.poly}\nLFSR length: {self.length}\
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
        return self.output
    
    def run_steps(self, N):
        print(f'\nHere are {N} iterations of the Shrinking Generator')
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
    