import os

path = "/HDD/backuped/Promotion_data/Postdoc/240822_mDOM_efficiency/pre_analysis/output_data/"
fnames = [path+fname for fname in os.listdir(path) if "_summary_data_with_trigger_loss.dat" in fname]

with open(path+"241014_summary_data_with_trigger_loss.dat", "a") as f:
    for fname in fnames:
            theta = fname.split("theta_")[1].split("_")[0]
            with open(fname, "r") as f2:
                for line in f2:
                    if "#" not in line:
                        f.write(theta + "\t")
                        f.write(line)

    for fname in fnames:
        os.remove(fname)

