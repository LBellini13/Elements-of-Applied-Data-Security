import numpy as np
import random

def ExtendedEuclideanAlgorithm(a, m):
    r = [m, a]
    s = [0, 1]
    t = [1, 0]
    q = [0, 0]
    i = 1
    while r[-1] != 0:
        i += 1
        r.append(r[i-2] % r[i-1])
        q.append((r[i-2] - r[i])/r[i-1])
        s.append(s[i-2] - q[i]*s[i-1])
        t.append(t[i-2] - q[i]*t[i-1])
    return r[i-1], s[i-1], t[i-1]

def int_to_bits(n):
    bits = []
    while n:
        bits.append(n & 1)
        n >>= 1
    return bits[::-1] if bits else [0]

def SquareAndMultiply(x, e, n):
    y = x
    for ebit in int_to_bits(e)[1:]:
        y = (y**2) % n
        if ebit:
            y = (y*x) % n
    return y

def findqr(p):
    '''
    Function to find q and r so that p = q*2^r + 1\n
    input:
        p-> number fro which compute q and r
    '''
    r = 0
    while (p - 1) % (2 ** (r + 1)) == 0:
        r += 1
    q = (p - 1) // (2 ** r)
    return q, r

def MillerRabin(p, N):
    q, r = findqr(p)
    print(f'q = {q}, r = {r}')
    for _ in range(N):
        # print('------------------')
        x = random.randint(2, p-2)
        # print(f'x = {x}')
        y = pow(x, q, p)
        # print(f'x^q\\p = {y}')
        if y == 1 or y == p-1:
            continue
        for _ in range(r):
            y = pow(y, 2, p)
            # print(f'y^2\\p = {y}')
            if y == p-1:
                break
        else:
            return False
    return True

class RSA:
    ''' Class implement RSA'''

    def __init__(self, length = None, n = None, e = None):
        self.length = length
        self.n = n
        self.e = e
    
    def modulus(self):
        n = 0 
        while n.bit_length() < self.length:
            p = random.getrandbits(self//2)
            q = random.getrandbits(self.length//2)
            n = p*q
        return p, q, n
    
    def modular_inverse(self, m):


    def encrypt(self, plaintext):

    def decrypt(self, ciphertext):
        ciphertext_int = int.from_bytes(ciphertext, byteorder='big')
        plaintext_int = ciphertext_int^self.d % self.n
        plaintext = plaintext_int.to_bytes((integer_value.bit_length() + 7) // 8, byteorder='big')
