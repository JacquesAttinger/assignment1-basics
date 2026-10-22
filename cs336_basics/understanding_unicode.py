test_string = "hello"
utf8_encoded = test_string.encode("utf-8")
print(utf8_encoded)

print(type(utf8_encoded))
for byte in utf8_encoded:
    print(byte)
    print(type(byte))

# bytestring = 
res = []
for b in utf8_encoded:
    res.append(bytes([b]).decode("utf-8"))
print(res)  
res_string = "".join(res)
print(res_string)
print('')

def decode_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])
print(chr(127))
print(decode_wrong(chr(128).encode("utf-8")))