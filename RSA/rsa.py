import numpy as np
import random
import math

def ExtendedEuclideanAlgorithm(a, m):
    '''
    Computes the gcd of a and m (supposing m > a) and s and t so that
    gcd(a, m) = sa + tm. If gcd(a, m) = 1, then s is the inverse of a with
    respect to multiplication modulo m.
    --------
    a: int, >= 0
    m: int, >= 0
        m > a
    --------
    tuple of int
        gcd(a, m), s and t so that sa + tm = gcd(a, m)
    '''
    # Ensure that m is bigger than a
    m, a = max(a, m), min(a, m)

    if a == 0:
        return m, 0, 1
    elif m == 0:
        return a, 1, 0
    
    r = [m, a]
    s = [0, 1]
    t = [1, 0]
    i = 1
    while r[-1] != 0:
        i += 1
        # Compute new quotient
        q = r[i-2] // r[i-1]
        # COmpute new remainder
        r.append(r[i-2] % r[i-1])
        # Update s and t
        s.append(s[i-2] - q*s[i-1])
        t.append(t[i-2] - q*t[i-1])
    
    # If s is negative and the gcd is 1, compute s+m to achieve the correct
    # modular inverse
    if s[i-1] < 0 and r[i-1] == 1:
        s[i-1] = m + s[i-1]

    return r[i-1], s[i-1], t[i-1]

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
    # print(f'n: {p}')

    # Manage cases for which x = random.randint(2, p-2) doesn't make sense   
    if p <= 1:
        return False
    if p == 2 or p == 3:
        return True
    if p % 2 == 0:
        return False
    
    q, r = find_q_r(p)
    # print(f'q: {q}, r: {r}')
    for _ in range(N):
        # print('------------------')
        x = random.randint(2, p-2)
        # print(f'x = {x}')
        y = SquareAndMultiply(x, q, p)
        # print(f'x^q\\p = {y}')
        if y == 1 or y == p-1:
            # print('AAAAAAAAAAAAAAAAAAAA')
            continue
        for _ in range(r):
            y = SquareAndMultiply(y, 2, p)
            # print(f'y^2\\p = {y}')
            if y == p-1:
                # print('BBBBBBBBBBBBBBBBBBBB')
                break
        else:
            # print('CCCCCCCCCCCCCC')
            return False
    return True

class RSA:
    '''
    Class implementing RSA\n
    --------
    Attributes:\n
        length: int (default = 512)
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
        _draw_random_prime_number:
            int
                random prime number tested with Miller Rabin Primality test
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
        do_encrypt = (self.debug and n is None and e is None) \
            or (not self.debug and length is not None)
        if do_encrypt:
            # print('ENCRYPT')
            if length is not None:
                self.length = length
            self.p, self.q, self.n, self.m = self._modulus()
            self.e, self.d = self._find_e_d()
            self.pub_key = (self.n, self.e)
            self.priv_key = (self.n, self.d)
        else:
            # print('DECRYPT')
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
                # print('DEBUG --> looking for p')
                p = self._draw_random_prime_number()
                # print('DEBUG --> looking for q')
                q = self._draw_random_prime_number()
                if p == q:
                    continue
                n = p*q
        else: # DEBUG branch
            p = 335895919357171
            q = 744053548667773
            n = p * q
        # print('DEBUG --> n computed')
        m = (p-1) * (q -1)
        # print('DEBUG --> m computed')
        return p, q, n, m
    
    def _draw_random_prime_number(self):
        n = random.getrandbits(self.length//2)
        # In the worst case scenario, the probability that Miller Rabin Test 
        # decalres as prime a composite number is 1/4, meaning that after k iterations
        # the probability of error is 4^(-k). 100 iterations are more than
        # sufficient for our application.
        while not MillerRabin(n, 100):
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
        # print('DEBUG --> looking for e and d')
        while  r != 1  and e < self.m:
            # Draw random e
            # e = np.random.randint(2, m - )
            e += 1  # e is alway sthe minimum. Done for repeatability
            r, d, _ = ExtendedEuclideanAlgorithm(e, self.m)
        # print('DEBUG --> e and d computed')
        return e, d

    def encrypt(self, plaintext):
        if not self.debug:
            plaintext_int = int.from_bytes(plaintext, byteorder='big')
            # print(f'pl int: {plaintext_int}')
            # print(f'n: {self.n}')
            if plaintext_int > self.n:
                raise ValueError('ERROR -> the plaintext to encrypt is too long. '\
                                 'Try with a shorter one.')
            # print('DEBUG --> encrypting')
            ciphertext_int = SquareAndMultiply(plaintext_int, self.e, self.n)
            ciphertext = ciphertext_int.to_bytes(math.ceil(
                ciphertext_int.bit_length()/8), byteorder='big')
        else: # DEBUG branch
            # print(f'pl int: {plaintext}')
            # print(f'n: {self.n}')
            if plaintext > self.n:
                raise ValueError('ERROR -> the number to encrypt is bigger than n. '\
                                 'Try with a smaller one.')
            ciphertext = SquareAndMultiply(plaintext, self.e, self.n)
        return ciphertext
    
    def decrypt(self, ciphertext):
        if not self.debug:
            ciphertext_int = int.from_bytes(ciphertext, byteorder='big')
            # print(f'cip int: {ciphertext_int}')
            # print(f'n dec: {self.n}')
            plaintext_int = SquareAndMultiply(ciphertext_int, self.d, self.n)
            # print(f'pl int dec: {plaintext_int}')
            # print('DEBUG --> decrypting')
            plaintext = plaintext_int.to_bytes(math.ceil(
                plaintext_int.bit_length()/8), byteorder='big')
        else: # DEBUG branch
            # print(f'cip: {ciphertext}')
            # print(f'n dec: {self.n}')
            plaintext = SquareAndMultiply(ciphertext, self.d, self.n)
        return plaintext
    
def ProbabilityToBePrime(L, threshold, init_iter, max_iter):
    '''
    Estimates the probability that an odd random integer in the interval 
    [2^L; 2^(L+1)] is prime exploting the Monte Carlo Method.
    --------
    L, int
        define the interval
    threshold, int
        minimum relative deviation between two consecutive estimations
    init_iter, int
        initial amount of iterations for the Monte Carlo simulation
    max_iter, int
        maximum number of iterations for the Monte Carlo simulation
    --------
    dict
        iterations and corresponding probability estimations
    float
        probability estimation
    '''
    lower_limit = 2**(L // 2)
    upper_limit = 2**(L // 2 +1)
    curr_prob, prev_prob = 0, 0
    estimations = {}
    prime_counter = 0
    iter = init_iter
    prev_rel_diff, curr_rel_diff = float('inf'), float('inf')
    while iter <= max_iter and (curr_rel_diff >= threshold or prev_rel_diff >= threshold):
        for _ in range(int(iter)):
            n = random.randint(lower_limit, upper_limit)
            # If the number is even add 1 and make it odd -> we don't lose
            # an iteration
            if n % 2 == 0:
                n += 1
            # In the worst case scenario, the probability that Miller Rabin Test 
            # decalres as prime a composite number is 1/4, meaning that after k iterations
            # the probability of error is 4^(-k). 100 iterations are more than
            # sufficient for our application.
            if MillerRabin(n, 100):
                prime_counter += 1

        prev_prob = curr_prob
        curr_prob = prime_counter/iter
        estimations[iter] = curr_prob
        prev_rel_diff= curr_rel_diff
        if prev_prob == 0:
            curr_rel_diff = float('inf')
        else:
            # Compute the relative difference between two consecutive
            # estimations
            curr_rel_diff = np.abs(curr_prob-prev_prob)/prev_prob
        # Double the number of iterations
        iter *= 2

    return estimations, curr_prob
    
