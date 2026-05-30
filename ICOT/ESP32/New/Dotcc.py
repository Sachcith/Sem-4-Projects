import numpy as np

mean_file = "scaler_mean.npy"
std_file  = "scaler_std.npy"

output_cc = "scaler.cc"

mean = np.load(mean_file)
std  = np.load(std_file)

with open(output_cc, "w") as f:
    f.write("#include <cstddef>\n\n")

    f.write(f"const size_t SCALER_SIZE = {len(mean)};\n\n")

    # Write mean array
    f.write("const float scaler_mean[] = {\n")
    for v in mean:
        f.write(f"    {float(v):.10f}f,\n")
    f.write("};\n\n")

    # Write std array
    f.write("const float scaler_std[] = {\n")
    for v in std:
        f.write(f"    {float(v):.10f}f,\n")
    f.write("};\n")

print(f"Generated {output_cc}")