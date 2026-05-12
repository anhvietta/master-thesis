"""
Run time comparison between ProtSearch, MMseqs2-GPU and CUDASW4++
"""

import matplotlib.pyplot as plt

xvals = [100000, 200000, 500000, 1000000, 2000000, 5000000]
cudasw_query_data = [3.40, 4.90, 12.19, 24.10, 48.51, 84.90]
cudasw_index_data = [0.43, 1.14, 2.72, 11.85, 21.545, 19.55]
mmseqs2_index_data = [2.99+6.03, 9.77+8.61, 11.58 +
                      17.25, 20.11+31.16, 39.89+56.27, 72.78+83.01]
mmseqs2_query_data = [11.524, 13.478, 17.084, 21.84, 30.642, 47.52]
protsearch_query_data = [3.01+4.87, 5.62+6.16, 14.1+9.10, 24.71+24.5]
protsearch_index_data = [127.271, 247.60, 600.771, 1172.73]
'''
fig, ax = plt.subplots()
ax.plot(xvals, cudasw_query_data, label="CUDASW4++")
ax.plot(xvals, mmseqs2_query_data, label="MMseqs2-GPU")
ax.plot(xvals[:len(protsearch_query_data)], protsearch_query_data, label="ProtSearch")
ax.set_xlabel("Log database size")
ax.set_ylabel("Time (s)")
ax.set_title("Query Time")
ax.set_xscale("log")
ax.legend()
plt.savefig("/scratch/stud2018/ata/measurements2/query_comp.png")
'''
fig, ax = plt.subplots()
ax.plot(xvals, cudasw_index_data, label="CUDASW4++")
ax.plot(xvals, mmseqs2_index_data, label="MMseqs2-GPU")
ax.plot(xvals[:len(protsearch_index_data)],
        protsearch_index_data, label="ProtSearch")
ax.set_xlabel("Log database size")
ax.set_ylabel("Log Time")
ax.set_title("Indexing Time")
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend()
plt.savefig("/scratch/stud2018/ata/measurements2/index_comp.png")
