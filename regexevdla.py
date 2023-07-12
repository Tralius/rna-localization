import re
pattern = re.compile(r"Epoch (\d+)/\d+\n\d+/\d+\w\s\[=+\]\s-.*val_loss:\s([\-\d\.]+)\s- val_ERM:\s([\-\d\.]+)\s- val_KDEL:\s")

# find all matches to groups
for match in pattern.finditer(a):
    # extract words
    print("Epoch: " + match.group(1))
    print("Val loss:" + match.group(2))
    print("")
    # extract numbers
    #print(match.group(2))
    print("")