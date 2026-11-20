"""Figure 6: how multiplying random factors gives a stretched exponential, with the numbers quoted in the post's illustration.

Each side of a box is a whole number >= 1, and every extra unit halves its chance, so a box l x w x h has chance
2^-(l+w+h-3) relative to the unit box. For a given volume V the most likely box is a cube of side V^(1/3), so the chance
of the best box of volume V is 2^(-3 (V^(1/3) - 1)): ln P is linear in V^(1/3), a stretched exponential with c = 1/3.
One factor alone has chance 2^-(V-1): a plain exponential.

Output: figures/fig6_mechanism.png (a plain rendering of the same arithmetic, not the post's illustration)
Run: python3 scripts/fig6_mechanism.py
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style
style.apply()
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

chance = lambda l, w, h: 2.0 ** -(l + w + h - 3)
print("Chance relative to a 1x1x1 box")
for box in ((3, 3, 3), (9, 3, 1), (27, 1, 1)):
    print(f"  {box[0]} x {box[1]} x {box[2]}: 1 in {1 / chance(*box):,.0f}")
print("  cube steps: volume gained", [k ** 3 - (k - 1) ** 3 for k in (2, 3, 4, 5)], "each for a factor of 8 in chance")
print(f"  one factor alone at volume 125: 1 in {2.0 ** 124:.2g}")

fig, a = plt.subplots(figsize=(6.4, 4.4))
V = np.arange(1, 126)
a.semilogy(V, 2.0 ** -(V - 1), color=style.PL, ls="--", label="one factor: chance halves per unit (exponential)")
a.semilogy(V, 2.0 ** (-3 * (V ** (1 / 3) - 1)), color=style.SE, label="three factors, best box is a cube (c = 1/3)")
a.scatter([8, 27, 64, 125], [1 / 8, 1 / 64, 1 / 512, 1 / 4096], color=style.SE, zorder=3, s=18)
a.set_xlabel("volume"); a.set_ylabel("chance relative to the unit box"); a.set_ylim(1e-38, 2)
a.set_title("Multiply a few random factors", loc="left"); a.legend(loc="lower left", fontsize=8.5)
fig.tight_layout(); fig.savefig(os.path.join(ROOT, "figures", "fig6_mechanism.png")); plt.close(fig)
print("\nwrote figures/fig6_mechanism.png")
