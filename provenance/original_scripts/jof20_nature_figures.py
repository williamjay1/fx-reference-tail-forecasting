"""Nature-inspired, data-faithful vector graphics for the existing JoF article."""
from pathlib import Path
import hashlib,json,sys
import fx_windows_platform_compat
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
from PIL import Image
import fitz

ROOT=Path('D:/MLWork/FXTailRisk');OUT=ROOT/'results/jof_v20_visual/figures';STYLE=OUT.parent/'style_sources'
OUT.mkdir(parents=True,exist_ok=True)
plt.style.use([str(STYLE/'science.mplstyle'),str(STYLE/'nature.mplstyle'),str(STYLE/'no-latex.mplstyle')])
plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.labelsize':8,'axes.titlesize':8,
    'xtick.labelsize':7.5,'ytick.labelsize':7.5,'legend.fontsize':7.5,'svg.fonttype':'none',
    'pdf.fonttype':42,'ps.fonttype':42,'text.usetex':False,'savefig.bbox':None,
    'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,
    'xtick.direction':'out','ytick.direction':'out','xtick.minor.visible':False,'ytick.minor.visible':False,
    'xtick.major.width':.6,'ytick.major.width':.6,'xtick.major.size':2.5,'ytick.major.size':2.5,'xtick.top':False,'ytick.right':False,
    'figure.facecolor':'white','axes.facecolor':'white','legend.frameon':False})
BLUE='#0072B2';ORANGE='#B55A00';GREY='#595959';TAILS=[(.95,BLUE,'-'),(.99,ORANGE,'--')]
BOOKS=[('funded_assets','Assets'),('net_cashflows','Flows')]
INPUTS={'dispersion':ROOT/'results/joint_fx_v18/inference/anchor_free_reference_daily.csv',
    'contrasts':ROOT/'results/joint_fx_20261002/analysis/primary_score_contrasts.csv',
    'pit':ROOT/'results/joint_fx_v18/pit_precision/full96/pit.csv',
    'predictions':ROOT/'results/joint_fx_20261002/internal_full/predictions.csv'}
MANIFEST=[]

def time_axis(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator(2));ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_xlim(pd.Timestamp('2018-01-01'),pd.Timestamp('2025-12-31'))

def panel(ax,letter,title):
    ax.text(-.14,1.12,letter,transform=ax.transAxes,ha='left',va='bottom',weight='bold',size=9)
    ax.set_title(title,loc='left',pad=8,weight='normal')

def audit(fig):
    fig.canvas.draw();r=fig.canvas.get_renderer();bounds=fig.bbox
    texts=[t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    # Tick labels outside the active (bottom/left) axes are not actually drawn.
    texts=[t for t in texts if t.get_window_extent(r).width>0 and t.get_window_extent(r).height>0]
    clipping=[];overlap=[];crossings=[]
    bbs=[t.get_window_extent(r) for t in texts]
    for t,b in zip(texts,bbs):
        if b.x0<bounds.x0-.5 or b.y0<bounds.y0-.5 or b.x1>bounds.x1+.5 or b.y1>bounds.y1+.5:clipping.append(t.get_text())
    for i in range(len(texts)):
        for j in range(i+1,len(texts)):
            b,c=bbs[i],bbs[j]
            if min(b.x1,c.x1)-max(b.x0,c.x0)>.5 and min(b.y1,c.y1)-max(b.y0,c.y0)>.5:
                overlap.append([texts[i].get_text(),texts[j].get_text()])
    for ax in fig.axes:
        for line in ax.lines:
            if not line.get_visible():continue
            path=line.get_transform().transform_path(line.get_path())
            for t,b in zip(texts,bbs):
                if path.intersects_bbox(b,filled=False):crossings.append(t.get_text())
    return {'clipped_text':clipping,'text_overlaps':overlap,'text_line_crossings':crossings,'text_items':len(texts)}

def export(fig,name,description,extra):
    qa=audit(fig)
    assert not qa['clipped_text'],(name,qa)
    assert not qa['text_overlaps'],(name,qa)
    assert not qa['text_line_crossings'],(name,qa)
    fig.savefig(OUT/(name+'.pdf'),facecolor='white',metadata={'Title':description,'Author':''})
    fig.savefig(OUT/(name+'.svg'),facecolor='white',metadata={'Title':description})
    fig.savefig(OUT/(name+'_1200dpi.png'),dpi=1200,facecolor='white')
    fig.savefig(OUT/(name+'_preview.png'),dpi=220,facecolor='white')
    # An explicit RGB high-resolution file is rendered, never resized from old PNGs.
    with Image.open(OUT/(name+'_1200dpi.png')) as im:
        pixel_size=list(im.size);dpi=list(im.info['dpi']);mode=im.mode
        if im.mode=='RGBA':im.convert('RGB').save(OUT/(name+'_1200dpi.png'),dpi=(1200,1200))
    svg=(OUT/(name+'.svg')).read_text(encoding='utf-8');assert '<image' not in svg and '<text' in svg
    with fitz.open(OUT/(name+'.pdf')) as pdf:
        assert not pdf[0].get_images(full=True),'Raster inside vector PDF'
        fonts=pdf[0].get_fonts();size=[float(pdf[0].rect.width)/72*25.4,float(pdf[0].rect.height)/72*25.4]
    MANIFEST.append({'name':name,'description':description,'physical_mm':size,'pixels':pixel_size,'dpi':dpi,
        'vector_has_raster':False,'svg_editable_text':True,'pdf_fonts':fonts,'layout_audit':qa,**extra})
    plt.close(fig)

def figure1():
    data=pd.read_csv(INPUTS['dispersion'],parse_dates=['origin_date'])
    data=data.query("study=='legacy_historical' and horizon==1").copy()
    tasks=[('funded_assets','EUR','Assets, EUR'),('funded_assets','USD','Assets, USD'),('net_cashflows','EUR','Flows, EUR'),('net_cashflows','USD','Flows, USD')]
    records=[];curves={}
    for book,report,title in tasks:
        for tau,_,_ in TAILS:
            sub=data[(data.book==book)&(data.report==report)&np.isclose(data.tau,tau)].sort_values('origin_date')
            assert len(sub)==2046 and sub.origin_date.is_unique
            values=sub.set_index('origin_date').anchor_symmetric_R_ES*100;roll=values.rolling(60,min_periods=30)
            z=pd.DataFrame({'origin_date':values.index,'median':roll.median().values,'p10':roll.quantile(.1).values,'p90':roll.quantile(.9).values,'book':book,'report':report,'tau':tau})
            curves[(book,report,tau)]=z;records.append(z)
    plotted=pd.concat(records,ignore_index=True);plotted.to_csv(OUT/'figure1_plot_data.csv',index=False)
    upper=np.ceil(plotted.p90.max()/10)*10
    fig,axes=plt.subplots(4,2,figsize=(180/25.4,168/25.4),sharex=True,sharey=True)
    fig.subplots_adjust(left=.105,right=.98,bottom=.085,top=.89,wspace=.20,hspace=.70)
    fig.text(.32,.976,'95% ES',ha='center',va='top',weight='bold',size=9)
    fig.text(.78,.976,'99% ES',ha='center',va='top',weight='bold',size=9)
    fig.legend([Line2D([],[],color=GREY,lw=1),Patch(facecolor=GREY,alpha=.12)],['Trailing 60-origin median','10th–90th percentiles (descriptive)'],loc='upper center',bbox_to_anchor=(.53,.958),ncol=2,handlelength=1.8,columnspacing=2)
    for row,(book,report,title) in enumerate(tasks):
        for col,(tau,color,ls) in enumerate(TAILS):
            ax=axes[row,col];z=curves[(book,report,tau)]
            ax.fill_between(z.origin_date,z.p10,z.p90,color=color,alpha=.13,lw=0)
            ax.plot(z.origin_date,z['median'],color=color,ls=ls,lw=1)
            panel(ax,chr(97+row*2+col),title)
            ax.set_ylim(0,upper);ax.set_yticks(np.arange(0,upper+1,20));ax.grid(axis='y',color='#dddddd',lw=.4)
            time_axis(ax)
            if col==0:ax.set_ylabel('ES span (%)')
            if row==3:ax.set_xlabel('Forecast origin')
    export(fig,'figure1_reference_sensitivity_nature','Reference ES forecast dispersion over time',{'origins_per_series':2046,'panels':8,'rolling_window':60,'minimum_observations':30,'band':'descriptive within-window 10th–90th percentiles','common_y_upper_percent':upper})

def figure2():
    df=pd.read_csv(INPUTS['contrasts']);assert len(df)==20
    assert (df.simultaneous_low<0).all() and (df.simultaneous_high>0).all()
    df.to_csv(OUT/'figure2_plot_data.csv',index=False)
    ids=['IN_FHS1250@EUR','FHS1250@EUR','EWMA_GAUSS','EWMA_SELFNORMALIZED','TAYLOR_OPT']
    names=['IN-FHS, EUR fit','Ordinary FHS, EUR fit','Gaussian EWMA','Internal elliptical EWMA','Taylor optimized']
    fig=plt.figure(figsize=(180/25.4,93/25.4))
    grid=fig.add_gridspec(1,3,left=.03,right=.985,bottom=.26,top=.78,width_ratios=[1.6,1.9,1.9],wspace=.16)
    labels=fig.add_subplot(grid[0,0]);labels.set_xlim(0,1);labels.set_ylim(-.65,4.65);labels.axis('off')
    for y,name in zip(range(4,-1,-1),names):labels.text(0,y,name,ha='left',va='center',size=7.5)
    fig.legend([Line2D([],[],color=BLUE,marker='o',lw=1,ms=3),Line2D([],[],color=ORANGE,marker='s',lw=1,ms=3)],['20-date blocks','60-date blocks'],loc='upper center',bbox_to_anchor=(.66,.975),ncol=2,handlelength=2,columnspacing=2)
    for col,tau in enumerate([.95,.99],1):
        ax=fig.add_subplot(grid[0,col]);ax.set_ylim(-.65,4.65)
        for idx,key in enumerate(ids):
            for block,color,marker,offset in [(20,BLUE,'o',.14),(60,ORANGE,'s',-.14)]:
                z=df[(df.comparator==key)&np.isclose(df.tau,tau)&(df.block==block)].iloc[0]
                ax.errorbar(z.delta,4-idx+offset,xerr=np.array([[z.delta-z.simultaneous_low],[z.simultaneous_high-z.delta]]),fmt=marker,ms=3,color=color,ecolor=color,lw=.9,capsize=2,capthick=.7)
        ax.axvline(0,color='#555555',lw=.6,ls=':');ax.set_yticks([]);ax.spines['left'].set_visible(False)
        panel(ax,chr(96+col),f'{tau:.0%} tail')
        if tau==.95:ax.set_xlim(-.045,.055);ax.set_xticks([-.04,-.02,0,.02,.04])
        else:ax.set_xlim(-.30,.065);ax.set_xticks([-.3,-.2,-.1,0])
        ax.ticklabel_format(style='plain',axis='x',useOffset=False)
        ax.set_xlabel('Mixture − comparator score')
    fig.text(.63,.058,'Negative values favor the mixture; panels use different x scales.',ha='center',va='center',size=7)
    export(fig,'figure2_score_contrasts_nature','Mixture score contrasts and simultaneous intervals',{'intervals':20,'family_size_per_tail':5,'resamples':5000,'all_intervals_include_zero':True,'different_tail_x_scales':True})

def figure3():
    pit=pd.read_csv(INPUTS['pit'],parse_dates=['origin_date'])
    pred=pd.read_csv(INPUTS['predictions'],parse_dates=['origin_date'],usecols=['origin_date','book','report','tau','model','strict_hit'])
    pred=pred[(pred.model=='IN_FHS_REF_POOL')&(pred.origin_date>=pd.Timestamp('2018-01-01'))]
    models=[('HS1250','Common HS',GREY,''),('IN_FHS1250_EUR','EUR IN-FHS',ORANGE,'//'),('IN_FHS_REF_POOL','Reference mixture',BLUE,'..')]
    hist=[];rolling=[]
    for book,_ in BOOKS:
        for model,label,color,hatch in models:
            vals=pit[(pit.book==book)&(pit.report=='EUR')&(pit.model==model)].pit
            assert len(vals)==2046 and vals.between(0,1).all()
            counts,edges=np.histogram(vals,bins=np.linspace(0,1,11));assert counts.sum()==2046
            for i,count in enumerate(counts):hist.append({'book':book,'model':model,'bin_left':edges[i],'bin_right':edges[i+1],'count':int(count),'fraction':count/len(vals),'n':len(vals)})
        for tau,_,_ in TAILS:
            z=pred[(pred.book==book)&(pred.report=='EUR')&np.isclose(pred.tau,tau)].sort_values('origin_date')
            assert len(z)==2046 and z.origin_date.is_unique
            rate=z.strict_hit.rolling(250,min_periods=250).mean()*100
            rolling.append(pd.DataFrame({'origin_date':z.origin_date,'book':book,'tau':tau,'rate':rate,'strict_hit':z.strict_hit}))
    hist=pd.DataFrame(hist);roll=pd.concat(rolling,ignore_index=True)
    hist.to_csv(OUT/'figure3_pit_plot_data.csv',index=False);roll.to_csv(OUT/'figure3_rolling_plot_data.csv',index=False)
    fig,axes=plt.subplots(2,2,figsize=(180/25.4,120/25.4))
    fig.subplots_adjust(left=.10,right=.98,bottom=.13,top=.80,wspace=.28,hspace=.54)
    fig.legend([Patch(facecolor='white',edgecolor=c,hatch=h,lw=.7) for _,_,c,h in models],[m[1] for m in models],loc='upper center',bbox_to_anchor=(.5,.99),ncol=3,columnspacing=1.8,handlelength=1.6)
    fig.legend([Line2D([],[],color=BLUE,lw=1),Line2D([],[],color=ORANGE,lw=1,ls='--'),Line2D([],[],color=GREY,lw=.6,ls=':')],['95% VaR','99% VaR','Nominal reference'],loc='upper center',bbox_to_anchor=(.71,.94),ncol=3,columnspacing=1.2,handlelength=1.7)
    ymax=np.ceil(max(.12,hist.fraction.max()+.005)*50)/50
    for row,(book,name) in enumerate(BOOKS):
        ax=axes[row,0]
        for k,(model,label,color,hatch) in enumerate(models):
            z=hist[(hist.book==book)&(hist.model==model)];center=(z.bin_left+z.bin_right)/2+(k-1)*.027
            ax.bar(center,z.fraction,width=.021,facecolor='white',edgecolor=color,hatch=hatch,lw=.7,zorder=3)
        ax.axhline(.1,color=GREY,lw=.6,ls=':',zorder=2)
        ax.set_xlim(0,1);ax.set_ylim(0,ymax);ax.set_xticks([0,.2,.4,.6,.8,1]);ax.set_yticks([0,.05,.10,.15] if ymax>=.15 else [0,.05,.10])
        ax.set_xlabel('Realized-loss PIT');ax.set_ylabel('Fraction per 0.1 bin');panel(ax,chr(97+row*2),name+', EUR · PIT')
        ax=axes[row,1]
        for tau,color,ls in TAILS:
            z=roll[(roll.book==book)&np.isclose(roll.tau,tau)]
            ax.plot(z.origin_date,z.rate,color=color,ls=ls,lw=1)
            ax.axhline((1-tau)*100,color=color,lw=.6,ls=':')
        ax.set_ylim(0,12);ax.set_yticks([0,4,8,12]);time_axis(ax);ax.set_xlabel('Forecast origin')
        ax.set_ylabel('Rolling breach rate (%)');ax.grid(axis='y',color='#dddddd',lw=.4)
        panel(ax,chr(98+row*2),name+', EUR · breaches')
    export(fig,'figure3_pit_calibration_nature','Predictive cash distributions and rolling tail calibration',{'panels':4,'pit_bins':np.linspace(0,1,11).tolist(),'pit_n_per_model_and_book':2046,'rolling_window':250,'strict_breach_definition':True,'pit_y_upper':ymax})

def main():
    figure1();figure2();figure3()
    versions={'python':sys.version,'matplotlib':matplotlib.__version__,'pandas':pd.__version__,'numpy':np.__version__}
    inputs={k:{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for k,p in INPUTS.items()}
    (OUT/'FIGURE_MANIFEST.json').write_text(json.dumps({'status':'3_native_vector_figures_and_1200dpi_rasters','versions':versions,'inputs':inputs,'figures':MANIFEST},indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'figures':len(MANIFEST),'text_overlap_count':0,'text_line_crossings':0,'output':str(OUT)}))

if __name__=='__main__':main()
