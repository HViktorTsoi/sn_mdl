# coding:utf-8
import base64
import json
import os
from Crypto.Cipher import AES
import codecs

class EncryptService:
    def __init__(self):
        self.modulus = '00e0b509f6259df8642dbc35662901477df22677ec152b5ff68ace615bb7b725152b3ab17a876aea8a5aa76d2e417629ec4ee341f56135fccf695280104e0312ecbda92557c93870114af6c9d05c4f7f0c3685b7a46bee255932575cce10b424d813cfe4875d3e82047b97ddef52741d546b8e289dc6935b3ece0462db0a22b8e7'
        self.nonce = '0CoJUm6Qyw8W8jud'
        self.pubKey = '010001'
        pass

    def __aesEncrypt(self, text, secKey):
        pad = 16 - len(text) % 16
        text = text + pad * chr(pad)
        encryptor = AES.new(secKey, 2, '0102030405060708')
        ciphertext = encryptor.encrypt(text)
        ciphertext = base64.b64encode(ciphertext)
        return ciphertext.decode("utf-8")  # VIP!!!!!!!!!!! 对于版本3.X必须进行decode

    def __rsaEncrypt(self, text, pubKey, modulus):
        text = text[::-1]
        # python3 这里要用codecs.encode()
        rs = int(codecs.encode(text.encode("utf-8"), "hex"), 16) ** int(pubKey, 16) % int(modulus, 16)
        return format(rs, 'x').zfill(256)

    def __createSecretKey(self, size):
        return (''.join(map(lambda xx: (hex(xx)[2:]), os.urandom(size))))[0:16]

    def encryptData(self, originData):
        data = json.dumps(originData)
        secKey = self.__createSecretKey(16)
        encText = self.__aesEncrypt(self.__aesEncrypt(data, self.nonce), secKey)
        encSecKey = self.__rsaEncrypt(secKey, self.pubKey, self.modulus)
        return {
            'params': encText,
            'encSecKey': encSecKey
        }