import re
pat = re.compile(r'\[[^\]]+\]|%\(\d+\)|%\d{2}|[0-9\.\(\)]|[^%0-9\.\(\)\[\]]+')
print(pat.findall("N%128.C%(101)2"))
