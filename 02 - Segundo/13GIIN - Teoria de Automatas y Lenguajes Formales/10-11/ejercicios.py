import re

# 1. Palabras que terminan siempre en `a`
p = re.compile("[a-b]*a")

# 2. Palabras que tienen un numero PAR de `a`
p = re.compile("b*ab*ab*")

# 3. Palabras que tienen un numero IMPAR de `a`
p = re.compile("b*a(b*ab*a)*")

# 4. Palabras donde `a` este precedida y sucedida por `b` y exista al menos una `a`
p = re.compile("bb*abb*(abb*)*")

# 5. No exista `aa` o `bb`
p = re.compile("a((ba)*b)|b((ab)*a)")
