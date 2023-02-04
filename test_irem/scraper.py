import re
files = ['slurm-467052.out']
all_details = []
for file in files:
    with open(file) as f:
        lines = f.readlines()
    errors = []
    gen_errors = []
    details = []
    for i in range(0, len(lines)):
        if lines[i].find("test error:") != -1:
            if lines[i].find("gen") != -1:
                num = re.findall(r'\d+',lines[i])
                number = '0.'+num[1]
                gen_errors.append(float(number))
            else:
                num = re.findall(r'\d+',lines[i])
                number = '0.'+num[1]
                errors.append(float(number))
        if lines[i].find("details:") != -1:
            details = lines[i][9:].strip().split(",")
    all_details.append([details, errors, gen_errors])
    
print(all_details)
