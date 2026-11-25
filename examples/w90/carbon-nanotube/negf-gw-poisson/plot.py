import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
R='/home/jiacao/quatrex_project/quatrex/examples/w90/carbon-nanotube/inputs/structure.xyz'
x=np.array([float(l.split()[1]) for l in open(R).readlines()[2:770]])
E=np.load('Z/electron_energies.npy').real; A=(np.load('Z/electron_density_47.npy')+np.load('Z/hole_density_47.npy')).real
p=np.load('Z/potential_23.npy').real; q=np.load('Z/real_space_charge_density_23.npy'); q=q[q!=0]
assert q.size==768 and abs(q.sum()-2.3933)<1e-3
a=102.8485/24; edges=np.arange(25)*a; mid=edges[:-1]+a/2
Ac=A.reshape(len(E),24,32).sum(2); pc=p.reshape(24,32).mean(1); qc=q.reshape(24,32).sum(1)
INK,INK2,MUT,GRID,SURF='#0b0b0b','#52514e','#898781','#e1e0d9','#fcfcfb'
BLUE,ORANGE='#2a78d6','#eb6834'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'text.color':INK,'axes.labelcolor':INK2,'xtick.color':INK2,'ytick.color':INK2,'axes.edgecolor':'#c3c2b7','axes.facecolor':SURF,'figure.facecolor':SURF,'axes.titlesize':10,'axes.titleweight':'bold','axes.titlelocation':'left'})
cm=LinearSegmentedColormap.from_list('seq',[SURF,'#cde2fb','#9ec5f4','#5598e7','#256abf','#184f95','#0d366b'])
fig,ax=plt.subplots(3,1,figsize=(8,10.5),sharex=True,gridspec_kw=dict(height_ratios=[2.6,1,1],hspace=.22),layout=None)
fig.subplots_adjust(left=.1,right=.87,top=.925,bottom=.06)
w=(E>=-4.5)&(E<=-2.2); e=E[w]; ee=np.r_[e-.0025,e[-1]+.0025]
m=ax[0].pcolormesh(edges,ee,np.clip(Ac[w],1e-2,None),norm=LogNorm(1e-2,1e2),cmap=cm,rasterized=True)
step=dict(where='post',lw=1.2)
for lev,lab in ((-3.080,'CBM'),(-3.735,'VBM')):
    ax[0].step(edges,np.r_[pc+lev,pc[-1]+lev],color=INK,ls=(0,(4,2)),**step)
    ax[0].text(1,pc[0]+lev+.02,f'flat-tube {lab} + mean potential',fontsize=7.5,color=INK)
ax[0].plot([0,8*a],[-2.896]*2,color=ORANGE,lw=2); ax[0].plot([16*a,24*a],[-2.996]*2,color=ORANGE,lw=2)
ax[0].text(.5,-2.885,'$E_F$ source −2.896',fontsize=8,color=INK,va='bottom'); ax[0].text(24*a-.5,-2.985,'$E_F$ drain −2.996',fontsize=8,color=INK,ha='right',va='bottom')
ax[0].set_ylim(-4.5,-2.2); ax[0].set_ylabel('Energy (eV)'); ax[0].set_title('LDOS, last SCBA iteration (GW + phonon)')
c=fig.add_axes([.885,ax[0].get_position().y0,.018,ax[0].get_position().height]); cb=fig.colorbar(m,cax=c,extend='min'); cb.set_label('States / (eV · unit cell)',color=INK2); cb.outline.set_visible(False)
ax[1].bar(mid,qc,width=a-.35,color=BLUE,label='Electrons − holes')
don=np.where((np.arange(24)<8)|(np.arange(24)>=16),.214,0.)
ax[1].plot([0,8*a],[.214]*2,color=ORANGE,lw=2,label='Donors (0.214)'); ax[1].plot([16*a,24*a],[.214]*2,color=ORANGE,lw=2)
ax[1].legend(frameon=False,fontsize=8,loc='upper center',labelcolor=INK2); ax[1].set_ylim(0,.25); ax[1].set_ylabel('Charge (per unit cell)'); ax[1].set_title('Excess electrons per unit cell')
ax[2].scatter(x,p,s=6,color=BLUE,alpha=.35,lw=0,label='Orbitals')
ax[2].plot(mid,pc,color=ORANGE,lw=2,marker='o',ms=4,label='Unit-cell mean')
ax[2].plot([51.424-13,51.424+13],[-.2]*2,color=INK,lw=3,solid_capstyle='butt'); ax[2].text(51.424,-.185,'gate (26 Å)',ha='center',fontsize=8,color=INK2)
ax[2].legend(frameon=False,fontsize=8,loc='upper right',labelcolor=INK2); ax[2].set_ylabel('Potential energy (eV)'); ax[2].set_title('Electrostatic potential energy, last SCSP step'); ax[2].set_xlabel('Position along the tube (Å)'); ax[2].set_xlim(0,24*a)
for k in ax[1:]:
    k.grid(axis='y',color=GRID,lw=.8); k.set_axisbelow(True)
for k in ax:
    for s in('top','right'): k.spines[s].set_visible(False)
    for b in(8*a,16*a): k.axvline(b,color=MUT,lw=.8,ls=':')
ax[0].text(12*a,-2.25,'channel (undoped)',ha='center',va='top',fontsize=8,color=INK2)
ax[0].text(4*a,-2.25,'n-doped lead',ha='center',va='top',fontsize=8,color=INK2); ax[0].text(20*a,-2.25,'n-doped lead',ha='center',va='top',fontsize=8,color=INK2)
fig.suptitle('CNT (8,0), 5 meV grid, default eta: V$_g$ = −0.3 V, V$_{ds}$ = 0.1 V',x=.1,y=.98,ha='left',fontsize=11,fontweight='bold')
fig.savefig('cnt_final.png',dpi=160); fig.savefig('cnt_final.pdf')
print('barrier',(pc.max()-pc[0]).round(4),'qc',qc.round(3),'sum',qc.sum().round(3),'donors',don.sum())
k=(e<-3.4)&(e>-3.9); i=np.argmax(Ac[w][k][:,11]); print('peak',e[k][i],Ac[w][k][i,11])
