from Crypto.Util.strxor import strxor

class AES_CBC():
    def __init__(self, k, iv):
        self.iv = iv
        self.cipher = AES.new(k, AES.MODE_ECB)
    def encrypt(self, msg):
        # Primero hacemos XOR del mensaje con el IV que tenemos
        m = strxor(msg, self.iv)
        c = self.cipher.encrypt(m)
        # Para la siguiente ronda, el IV es el propio texto cifrado
        self.iv = c
        return c
    def decrypt(self, encrypted_msg):
        out = self.cipher.decrypt(encrypted_msg)
        c = strxor(out, self.iv)
        self.iv = c
        return c


m = b'abcdefghabcdefgh'
print("Mensaje en texto plano: {}".format(m.decode("utf-8")))

k = get_random_bytes(16)
iv = get_random_bytes(16)
mycbc = AES_CBC(k, iv)

m1 = mycbc.encrypt(m)
print("Mensaje encriptado m1: {}".format(b64encode(m1)))

mycbcdecrypt = AES_CBC(k, iv)
print("Mensaje desencriptado m1: {}".format(mycbcdecrypt.decrypt(m1)))
