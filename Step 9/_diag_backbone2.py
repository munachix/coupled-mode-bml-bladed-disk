import sys, math
import numpy as np
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 6')
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 4')
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project')
import step6 as s6
import step4 as s4

inp = s6.load_inputs()
MODE = 2
K = inp['K_sec'][MODE, MODE]; M = inp['M_sec'][MODE, MODE]; C = inp['C_sec'][MODE, MODE]
K3 = inp['K3_sec_diag'][MODE]
q_ref = 1.0
omega0 = math.sqrt(K/M)

s4.CONFIG['continuation']['w_stop_hi'] = 3.0
s4.CONFIG['continuation']['n_steps'] = 4000

tp = 0.3
cont = s4.duffing_forced_response_continuation(omega0, M, C, K, K3, q_ref, tp)
f = cont['Omega']/(2*math.pi)
amp = cont['amplitude']
stable = cont['stable']
print("idx, f(Hz), amp(mm), stable  -- sampled every 10 points")
for i in range(0, len(f), 10):
    print(f"{i:4d}  {f[i]:8.2f}  {amp[i]:8.5f}  {stable[i]}")
print("LAST:", len(f)-1, f[-1], amp[-1], stable[-1])
