import math

a = "1E10"
find_e = "e" in a.lower()
print(a.index("e"))
exp_scientific = (
    int(a[a.index("e") + 1 :])
    if find_e
    else int(math.floor(math.log10(abs(float(a)))))
)
print(exp_scientific)