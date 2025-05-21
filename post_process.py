import os


if __name__ == "__main__":
	with open("new.csv", "r") as f:
		lines = f.readlines()	
	
	out_csv = open("post_process.csv", "w" )
	out_csv.write("id,label\n")
	
	lines = lines[1::]
	for line in lines:
		splited_line = line.split(",")
		prop = float(splited_line[2])
		if prop < 0.2:
			splited_line[1] = "14"
		
		to_write = ",".join(splited_line[:2])
		out_csv.write(f'{to_write}\n')

	out_csv.close()
