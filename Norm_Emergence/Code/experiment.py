my_string = "2019	882	4885	383	3519	431	5000	3792	732	2530	926	2494	1471	2577	5000	900	878	5000	824	520"
output = my_string.split(" ")
output_string = ""
for i in output:
    output_string += chr(int(i)) + ","
    
print(output_string)