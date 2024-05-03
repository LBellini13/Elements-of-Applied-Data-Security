import numpy as np
from matplotlib.image import imsave
import matplotlib.pyplot as plt
from Crypto.Cipher import AES

def aes_image_encryption(aes_cipher, mode, image):
    # Convert the matrix into a 1D vector and then convert into bytes
    image_bytes = bytes(image.flatten())
    # Encrypt using AES
    cipher_image_bytes = aes_cipher.encrypt(image_bytes)
    # Convert it back to 1D array of int
    cipher_image = [byte for byte in cipher_image_bytes]
    # Convert it back to a matrix of the same dimensions of the original image
    cipher_image = np.array(cipher_image).reshape(image.shape)
    plt.imshow(cipher_image, cmap='gray')
    # Save the image
    imsave(f'image_{mode}.png', cipher_image, cmap='gray')

def flip_bit(text, ibit):
    # Convert the input into bytearray format (bytes is immutable)
    flipped_text = bytearray(text) 
    # Flip the bit
    flipped_text[ibit//8] ^= 1 << (ibit%8) 
    return bytes(flipped_text)

def hamming(textA, textB):
    # Compute the bitwise XOR
    AxorB = bytes(a ^ b for (a, b) in zip(textA, textB))
    # Count how many 1s are in the sequence
    distance = sum([bin(byte).count('1') for byte in AxorB])
    return distance

def mcs_diffusion(cipher, ref_plaintext, ref_ciphertext, it):  
    dist = []
    for _ in range(it):
        # Flip one random bit of the plaintext
        plaintext = flip_bit(ref_plaintext, np.random.randint(8*len(ref_plaintext)))
        # Encrypt the modified plaintext
        ciphertext = cipher.encrypt(plaintext)
        # Compute the hamming distance between the original ciphertext and the new one
        dist.append(hamming(ref_ciphertext, ciphertext)/8/len(ref_ciphertext)*100)
    return dist

def mcs_confusion(cipher_type, ref_key, ref_plaintext, ref_ciphertext, it, drop = None):  
    dist = []
    # Check which type of cipher must be used
    if cipher_type == 'aes':
        for _ in range(it):
            # Flip one random bit of the key
            key = flip_bit(ref_key, np.random.randint(8*len(ref_key)))
            # Instantiate a new cipher with the modified key
            aes = AES.new(key, AES.MODE_ECB)
            # Encrypt the plaintext using the new key
            ciphertext = aes.encrypt(ref_plaintext)
            # Compute the hamming distance between the original ciphertext and the new one
            dist.append(hamming(ref_ciphertext, ciphertext)/8/len(ref_ciphertext)*100)
    elif cipher_type == 'rc4':
        for _ in range(it):
            # Flip one random bit of the key
            key = flip_bit(ref_key, np.random.randint(8*len(ref_key)))
            # Instantiate a new cipher with the modified key
            rc4 = RC4(key, drop)
            # Encrypt the plaintext using the new key
            ciphertext = rc4.encrypt(ref_plaintext)
            # Compute the hamming distance between the original ciphertext and the new one
            dist.append(hamming(ref_ciphertext, ciphertext)/8/len(ref_ciphertext)*100)
    return dist


class RC4():
    def __init__(self, key, drop = None):
        self.key = key
        self.drop = drop
        self.dropped_bytes = 0
        # Index pointers
        self.i, self.j = 0, 0
        # Initialize the secret permutation using KSA
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
        # Compute new iterations and discard first bytes (self.out is not updated)
        if self.drop is not None:
            while self.dropped_bytes < self.drop:
                self.prga()
                self.dropped_bytes += 1

    def __iter__(self):
        return self

    def __next__(self):
        # Drop bytes
        self.drop_bytes()
        self.prga()
        # COnvert from int to bytes
        self.out = self.out_int.to_bytes((self.out_int // 256 ) + 1, byteorder='big')
        return self.out
    
    def run_steps(self, n):
        keystream = []
        # Drop bytes
        self.drop_bytes()
        for _ in range(n):
            self.prga()
            self.out = self.out_int.to_bytes((self.out_int // 256 ) + 1, byteorder='big')
            keystream.append(self.out_int)
        return keystream
    
    def encrypt(self, plaintext):
        # Produce a keystream as long as the plaintext
        keystream = self.run_steps(len(plaintext))
        # Bitwise XOR between keystream and plaintext
        ciphertext = bytes([pb ^ kb for pb, kb in zip(plaintext, keystream)])
        return ciphertext
    
    def decrypt(self, ciphertext):
        # Produce a keystream as long as the plaintext
        keystream = self.run_steps(len(ciphertext))
        # Bitwise XOR between keystream and ciphertext
        plaintext = bytes([cb ^ kb for cb, kb in zip(ciphertext, keystream)])
        return plaintext
    
