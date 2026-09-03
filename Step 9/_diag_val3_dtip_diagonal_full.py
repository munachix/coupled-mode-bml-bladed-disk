import sys, json
import numpy as np
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 7')
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 4')
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project')
import step7 as s7

with open(r"Step 9\output\validation3_real_ansys_health_id.json") as f:
    r = json.load(f)
y_real = np.array(r['freqs_full_matched'])
TRUE_BLADE = r['damaged_blade_true']

inp = s7.load_inputs()
loc = s7.sparse_localize_blade(y_real, np.zeros(24), inp)
order = np.argsort(loc['residuals'])
print("=== Full 24-candidate DIAGONAL localization breakdown (d_tip case) ===")
for b in range(24):
    marker = " <-- TRUE" if b == TRUE_BLADE else ""
    print(f"  blade {b:2d}: severity={loc['severities'][b]:+.4f}  residual={loc['residuals'][b]:.3f}{marker}")
print(f"\nTrue blade rank: {list(order).index(TRUE_BLADE)+1} of 24, residual={loc['residuals'][TRUE_BLADE]:.3f}")
print(f"Best blade {order[0]}: residual={loc['residuals'][order[0]]:.3f}")
print(f"Residual spread: min={loc['residuals'].min():.3f} max={loc['residuals'].max():.3f} (ratio={loc['residuals'].max()/loc['residuals'].min():.2f}x)")
