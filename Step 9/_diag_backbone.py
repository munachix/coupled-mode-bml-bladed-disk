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
zeta = C/(2*math.sqrt(K*M))
kappa = 0.75*K3*q_ref**2/K
print(f"f0={omega0/2/math.pi:.3f} Hz  zeta={zeta:.5f}  kappa={kappa:.4f}")

s4.CONFIG['continuation']['w_stop_hi'] = 3.0
s4.CONFIG['continuation']['n_steps'] = 4000

for tp in [0.3, 0.5, 0.7, 1.0, 1.5]:
    cont = s4.duffing_forced_response_continuation(omega0, M, C, K, K3, q_ref, tp)
    w = cont['Omega']/omega0
    f = cont['Omega']/(2*math.pi)
    amp = cont['amplitude']
    stable = cont['stable']
    nfold = cont['n_folds']
    fidx = cont['fold_indices']
    print(f"\ntp={tp}: n_folds={nfold} n_points={len(w)} w_range=({w.min():.3f},{w.max():.3f})")
    if nfold >= 2:
        for k in fidx[:4]:
            print(f"   fold at index {k}: f={f[k]:.2f} Hz, amp={amp[k]:.4f} mm, w={w[k]:.4f}")
    # print stable-segment transitions
    trans = np.where(np.diff(stable.astype(int)) != 0)[0]
    print(f"   stable-mask transitions at indices: {trans[:10]}")
    for t in trans[:6]:
        print(f"     idx {t}->{t+1}: f {f[t]:.2f}->{f[t+1]:.2f} Hz, amp {amp[t]:.4f}->{amp[t+1]:.4f}, stable {stable[t]}->{stable[t+1]}")
    print(f"   amp at w=1.6 (BPINN cutoff, f={1.6*omega0/2/math.pi:.1f}Hz): idx={np.argmin(np.abs(w-1.6))}, amp={amp[np.argmin(np.abs(w-1.6))]:.5f}")
    print(f"   final amp (last point, f={f[-1]:.1f}Hz): {amp[-1]:.6f}")
