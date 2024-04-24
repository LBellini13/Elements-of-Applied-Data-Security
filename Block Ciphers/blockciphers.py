import numpy as np
from matplotlib.image import imread, imsave
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

def aes_mcs_diffusion(aes, ref_plaintext, ref_ciphertext, it):  
    dist = []
    for _ in range(it):
        plaintext = flip_bit(ref_plaintext, np.random.randint(8*AES.block_size))
        ciphertext = aes.encrypt(plaintext)
        dist.append(hamming(ref_ciphertext, ciphertext)/8/AES.block_size*100)
    return dist

def aes_mcs_confusion(ref_key, key_length, ref_plaintext, ref_ciphertext, it):  
    dist = []
    for _ in range(it):
        key = flip_bit(ref_key, np.random.randint(8*key_length))
        aes = AES.new(key, AES.MODE_ECB)
        ciphertext = aes.encrypt(ref_plaintext)
        dist.append(hamming(ref_ciphertext, ciphertext)/8/AES.block_size*100)
    return dist