import re

with open("D:\python\2023090910010-李骁涵-机器学习\secret.daz", 'r') as f:
    content = f.read()

hex_values = re.split('X+', content)

decrypted_text = ''.join([chr(int(x, 16)) for x in hex_values if x])

visible_chars = sum([1 for c in decrypted_text if c.isprintable() and not c.isspace()])

with open('interpretation.txt', 'w', encoding='utf-8') as f:
    f.write(decrypted_text)
    f.write('\n<解密人>1234567<情报总字数>{}'.format(visible_chars))