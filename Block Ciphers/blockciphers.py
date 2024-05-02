import numpy as np
from matplotlib.image import imsave
import matplotlib.pyplot as plt
from Crypto.Cipher import AES

def aes_image_encryption(aes_cipher, mode, image):
    image_bytes = bytes(image.flatten())
    cipher_image_bytes = aes_cipher.encrypt(image_bytes)
    cipher_image = [byte for byte in cipher_image_bytes]
    cipher_image = np.array(cipher_image).reshape(image.shape)
    plt.imshow(cipher_image, cmap='gray')
    imsave(f'image_{mode}.png', cipher_image, cmap='gray')

def flip_bit(text, ibit):
    flipped_text = bytearray(text) 
    flipped_text[ibit//8] ^= 1 << (ibit%8) 
    return bytes(flipped_text)

def hamming(textA, textB):
    AxorB = bytes(a ^ b for (a, b) in zip(textA, textB))
    distance = sum([bin(byte).count('1') for byte in AxorB])
    return distance

def mcs_diffusion(cipher, ref_plaintext, ref_ciphertext, it):  
    dist = []
    for _ in range(it):
        plaintext = flip_bit(ref_plaintext, np.random.randint(8*len(ref_plaintext)))
        ciphertext = cipher.encrypt(plaintext)
        dist.append(hamming(ref_ciphertext, ciphertext)/8/len(ref_ciphertext)*100)
    return dist

def mcs_confusion(cipher_type, ref_key, ref_plaintext, ref_ciphertext, it, drop = None):  
    dist = []
    if cipher_type == 'aes':
        for _ in range(it):
            key = flip_bit(ref_key, np.random.randint(8*len(ref_key)))
            aes = AES.new(key, AES.MODE_ECB)
            ciphertext = aes.encrypt(ref_plaintext)
            dist.append(hamming(ref_ciphertext, ciphertext)/8/len(ref_ciphertext)*100)
    elif cipher_type == 'rc4':
        for _ in range(it):
            key = flip_bit(ref_key, np.random.randint(8*len(ref_key)))
            rc4 = RC4(key, drop)
            ciphertext = rc4.encrypt(ref_plaintext)
            dist.append(hamming(ref_ciphertext, ciphertext)/8/len(ref_ciphertext)*100)
    return dist


class RC4():
    def __init__(self, key, drop = None):
        self.key = key
        self.drop = drop
        self.dropped_bytes = 0
        self.i, self.j = 0, 0
        self.p = self.KSA()
        self.out = None
        self.out_int = None

    def KSA(self):
        p = list(range(256))
        j = 0
        for i in range(256):
            j = (j + p[i] + self.key[i % len(self.key)]) % 256
            p[i], p[j] = p[j], p[i]
        return p

    def prga(self):
        self.i = (self.i + 1) % 256
        self.j = (self.j + self.p[self.i]) % 256
        self.p[self.i], self.p[self.j] = self.p[self.j], self.p[self.i]
        self.out_int = self.p[(self.p[self.i] + self.p[self.j]) % 256]

    def drop_bytes(self):
        if self.drop is not None:
            while self.dropped_bytes < self.drop:
                self.prga()
                self.dropped_bytes += 1

    def __iter__(self):
        return self

    def __next__(self):
        self.drop_bytes()
        self.prga()
        self.out = self.out_int.to_bytes((self.out_int // 256 ) + 1, byteorder='big')
        return self.out
    
    def run_steps(self, n):
        keystream = []
        self.drop_bytes()
        for _ in range(n):
            self.prga()
            self.out = self.out_int.to_bytes((self.out_int // 256 ) + 1, byteorder='big')
            keystream.append(self.out_int)
        return keystream
    
    def encrypt(self, plaintext):
        keystream = self.run_steps(len(plaintext))
        ciphertext = bytes([pb ^ kb for pb, kb in zip(plaintext, keystream)])
        return ciphertext
    
    def decrypt(self, ciphertext):
        keystream = self.run_steps(len(ciphertext))
        plaintext = bytes([cb ^ kb for cb, kb in zip(ciphertext, keystream)])
        return plaintext
    
