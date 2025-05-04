import json,matplotlib;matplotlib.use('TkAgg');import matplotlib.pyplot as plt,mplcursors
f='/Users/d/desk/b/bunny.json'
c=[i["center"]for i in json.load(open(f))if"center"in i]
plt.figure(figsize=(8,6))
cursor=mplcursors.cursor(plt.scatter([p[0]for p in c],[p[1]for p in c],c='blue',marker='o',s=100,linewidths=2),hover=1)
plt.xlabel("X");plt.ylabel("Y");plt.title("Centers Plot");plt.grid(1)
@cursor.connect("add")
def _(sel):p=c[sel.index];sel.annotation.set(text=f"({p[0]:.3f},{p[1]:.3f},{p[2]:.3f})")
plt.savefig("image.png")
plt.show()
#:<
