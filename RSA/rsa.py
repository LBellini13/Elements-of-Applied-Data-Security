import numpy as np
import random
import math

def ExtendedEuclideanAlgorithm(a, m):
    '''
    Computes the gcd of a and m (supposing m > a) and s and t so that
    gcd(a, m) = sa + tm. If gcd(a, m) = 1, then s is the inverse of a with
    respect to multiplication modulo m.
    --------
    a: int
    m: int
        m > a
    --------
    tuple of int
        gcd(a, m), s and t so that sa + tm = gcd(a, m)
        
    '''

    # (r0, r1) = (m, a)
    # (t0, t1, s0, s1) = (1, 0, 0, 1)

    # while (r0 != 0):
    #     (q, r1, r0) = (r1 // r0, r0, r1 % r0)
    #     (t0, t1) = (t1, t0 - q * t1)
    #     (s0, s1) = (s1, s0 - q * s1)

    # if(t0 < 0):
    #     t0 = t0 + m
    # return r1, t0, s0

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
    if s[i-1] < 0:
        s[i-1] = m + s[i-1]
    return r[i-1], int(s[i-1]), int(t[i-1])

def int_to_bits(n):
    bits = []
    while n:
        bits.append(n & 1)
        n >>= 1
    return bits[::-1] if bits else [0]

def SquareAndMultiply(base, exp, mod):
    '''
    Computes the exponentiation base^exp % mod by
    means of squaring and multiplication\n
    --------
    base: int
        power's base
    exp: int
        power's exponent
    mod: int
        modulus' value
    --------
    int
        result of the operation
    '''
    y = base
    for ebit in int_to_bits(exp)[1:]:
        y = (y**2) % mod
        if ebit:
            y = (y*base) % mod
    return y


def find_q_r(p):
    '''
    Finds q and r so that p = q*2^r + 1\n
    --------
    p: int
        number from which q and r are computed
    --------
    tuple of int
        q and r
    '''
    r = 0
    while (p - 1) % (2 ** (r + 1)) == 0:
        r += 1
    q = (p - 1) // (2 ** r)
    return q, r

def MillerRabin(p, N):
    '''
    Checks whether an input number is likely prime or surely composite\n
    --------
    p: int
        number to evaluate
    N: int
        number if iterations
    --------
    bool
        True -> p is likely prime\n
        False-> p is surely composite
    '''
    if p % 2 == 0:
        return False
    q, r = find_q_r(p)
    for _ in range(N):
        # print('------------------')
        x = random.randint(2, p-2)
        # print(f'x = {x}')
        y = SquareAndMultiply(x, q, p)
        # print(f'x^q\\p = {y}')
        if y == 1 or y == p-1:
            continue
        for _ in range(r):
            y = SquareAndMultiply(y, 2, p)
            # print(f'y^2\\p = {y}')
            if y == p-1:
                break
        else:
            return False
    return True

class RSA:
    '''
    Class implementing RSA\n
    --------
    Attributes:\n
        length: int (default = None)
            modulus' desired number of bits
        n: int (default = None)
            RSA modulus. 1st element of the public key
        e: int (default = None)
            number that is coprime with phi(n). 2nd element of the public key
    --------
    Methods:
    --------
        _modulus:
            tuple of int
                p and q (prime numbers) and n=p*q\n
    --------    
        _find_e_d:
            tuple of int
                e -> gcd(e, m) = 1\n
                d-> ed=1 (mod m)
    --------
        encrypt:
            applies RSA encryption function
                plaintext: bytes
                    plaintext to be encrypted
                bytes:
                    generated ciphertext
    --------
        decrypt:
            applies RSA ecryption function
                ciphertext: bytes
                    ciphertext to be decrypted
                bytes:
                    generated plaintext
    '''

    def __init__(self, length = None, n = None, e = None, debug = False):
        self.debug = debug
        if length is not None:
            self.length = length
            self.p, self.q, self.n, self.m = self._modulus()
            self.e, self.d = self._find_e_d()
            self.pub_key = (self.n, self.e)
            self.priv_key = (self.n, self.d)
        else:
            self.n = n
            self.e = e
            self.pub_key = (self.n, self.e)
    
    def _modulus(self):
        '''
        Generates two big random integer numbers p and q, 
        verifies if they are prime by means of Miller Rabin primality test. 
        Eventually computes n = p*q
        --------
        tuple of int
            p, q, and n
        '''
        if not self.debug:
            n = 0 
            while n.bit_length() != self.length:
                p = self.draw_random_prime_number()
                q = self.draw_random_prime_number()
                if p == q:
                    continue
        else: # DEBUG branch
            p = 0x1083e935648922e73
            q = 0x1cc1a881e36821695
        n = p * q
        m = (p-1) * (q -1)
        return p, q, n, m
    
    def draw_random_prime_number(self):
        n = random.getrandbits(self.length//2)
        while not MillerRabin(n, 500):
            n = random.getrandbits(self.length//2)
        return n
    
    def _find_e_d(self):
        '''
        Finds e and d so that gcd(e, m)=1 and ed=1 (mod m)\n
        --------
        tuple of int
            e and d
        '''
        r = 0
        e = 1
        while  r != 1  and e < self.m:
            # Draw random e
            # e = np.random.randint(2, m - )
            e += 1  # e is alway sthe minimum. Done for repeatability
            r, d, _ = ExtendedEuclideanAlgorithm(e, self.m)
        return e, d

    def encrypt(self, plaintext):
        if not self.debug:
            plaintext_int = int.from_bytes(plaintext, byteorder='big')
            print(f'plaintext in int format: {plaintext_int}')
            ciphertext_int = SquareAndMultiply(plaintext_int, self.e, self.n)
            ciphertext = ciphertext_int.to_bytes(math.ceil(
                ciphertext_int.bit_length()/8), byteorder='big')
        else: # DEBUG branch
            ciphertext = SquareAndMultiply(plaintext, self.e, self.n)
        return ciphertext
    
    def decrypt(self, ciphertext):
        if not self.debug:
            ciphertext_int = int.from_bytes(ciphertext, byteorder='big')
            plaintext_int = SquareAndMultiply(ciphertext_int, self.d, self.n)
            plaintext = plaintext_int.to_bytes(math.ceil(
                plaintext_int.bit_length()/8), byteorder='big')
        else: # DEBUG branch
            plaintext = SquareAndMultiply(ciphertext, self.d, self.n)
        return plaintext
