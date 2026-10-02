"""Native vector table plates, exact source cells, and high-resolution exports."""
from pathlib import Path
import csv, hashlib, json, re, sys
import fx_windows_platform_compat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from PIL import Image
import fitz

ROOT=Path('D:/MLWork/FXTailRisk')
OUT=ROOT/'results/jof_v20_visual/tables'
STYLES=ROOT/'results/jof_v20_visual/style_sources'
MM=1/25.4; WIDTH=180*MM; PAGE_PT=WIDTH*72
FONT=FontProperties(family='Arial',size=8)
BOLD=FontProperties(family='Arial',size=8,weight='bold')

def plain(value):
    return re.sub(r'\*\*([^*]*)\*\*',r'\1',value)

def tables(source):
    lines=source.read_text(encoding='utf-8').splitlines(); found=[]; caption='';i=0
    while i<len(lines):
        line=lines[i]
        if line.startswith('**Table '):caption=line
        if line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].startswith('|'):
                cells=[x.strip() for x in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?',x) for x in cells): rows.append(cells)
                i+=1
            assert rows and all(len(r)==len(rows[0]) for r in rows)
            found.append((caption,rows));continue
        i+=1
    return found

def measure(text,renderer,bold=False):
    return renderer.get_text_width_height_descent(text,BOLD if bold else FONT,ismath=False)[0]*72/100

def wrap(text,width,renderer,bold=False):
    words=plain(text).split(' '); lines=[];current=''
    for word in words:
        proposed=(current+' '+word).strip()
        if measure(proposed,renderer,bold)<=width:
            current=proposed;continue
        if current:lines.append(current);current=''
        while measure(word,renderer,bold)>width:
            n=1
            while n<len(word) and measure(word[:n+1],renderer,bold)<=width:n+=1
            lines.append(word[:n]);word=word[n:]
        current=word
    if current:lines.append(current)
    return '\n'.join(lines)

def widths(rows,renderer):
    n=len(rows[0]); cols=list(zip(*rows)); text_cols=[]
    for j,values in enumerate(cols):
        if any(re.search(r'[A-Za-z]',plain(v)) for v in values[1:]):text_cols.append(j)
    values=[max(measure(plain(v),renderer) for v in c[1:])+12 for c in cols]
    if n==9: fractions=[.30]+[.70/8]*8
    elif n==8: fractions=[.20,.10,.075,.10,.08,.125,.115,.205]
    elif n==6 and 'Comparator' in rows[0][0]:fractions=[.34,.07,.1475,.1475,.1475,.1475]
    elif n==6 and 'Forecast system' in rows[0][0]:fractions=[.35,.07,.145,.145,.145,.145]
    elif n==6 and 'Fixed origin' in rows[0][0]:fractions=[.16,.21,.07,.13,.23,.20]
    elif n==6 and 'Fitting reference' in rows[0][0]:fractions=[.24,.19,.13,.13,.20,.11]
    elif n==5 and ('Forecast system' in rows[0][0] or 'Configuration' in rows[0][0]):fractions=[.43,.15,.15,.135,.135]
    elif n==5 and 'Comparator' in rows[0][0]:fractions=[.32,.10,.24,.10,.24]
    elif n==5 and 'Reporting-currency share' in rows[0]:fractions=[.26,.08,.19,.28,.19]
    elif n==4 and rows[0][0]=='Tail' and rows[0][2].startswith('Smallest numerical'):fractions=[.08,.30,.27,.35]
    elif n==4 and rows[0][0]=='Tail':fractions=[.08,.38,.12,.42]
    else:
        for j in range(n):
            if j in text_cols: values[j]=min(max(values[j],80),170)
            else:values[j]=max(values[j],43)
        fractions=[x/sum(values) for x in values]
    total=PAGE_PT-16
    return [total*x/sum(fractions) for x in fractions]

def render_table(caption,rows,stem):
    probe=plt.figure(figsize=(WIDTH,1),dpi=100);probe.canvas.draw(); renderer=probe.canvas.get_renderer()
    colw=widths(rows,renderer)
    wrapped=[[wrap(v,colw[j]-12,renderer,bold=(i==0 or '**' in v)) for j,v in enumerate(row)] for i,row in enumerate(rows)]
    heights=[max(v.count('\n')+1 for v in row)*11+9 for row in wrapped]
    plt.close(probe)
    title=plain(caption).split('.**')[0]
    title=plain(caption)
    title_lines=wrap(title,PAGE_PT-16,renderer,bold=True).split('\n')
    title_h=14*len(title_lines)+14
    header_h=heights[0]; maxbody=670-title_h-header_h-38
    groups=[]; begin=1
    while begin<len(rows):
        end=begin;used=0
        while end<len(rows) and used+heights[end]<=maxbody:
            used+=heights[end];end+=1
        assert end>begin,'A single table row is taller than a plate'
        groups.append((begin,end));begin=end
    plates=[]
    for part,(begin,end) in enumerate(groups,1):
        row_ids=[0]+list(range(begin,end))
        height_pt=title_h+sum(heights[k] for k in row_ids)+14
        fig=plt.figure(figsize=(WIDTH,height_pt/72),dpi=100)
        ax=fig.add_axes([0,0,1,1]);ax.set_xlim(0,PAGE_PT);ax.set_ylim(height_pt,0);ax.axis('off')
        texts=[]; lines=[]
        title_suffix=f'  (continued {part}/{len(groups)})' if len(groups)>1 else ''
        rendered_title=title+title_suffix
        rendered_title=wrap(rendered_title,PAGE_PT-16,renderer,bold=True)
        texts.append(ax.text(8,8,rendered_title,fontproperties=BOLD,va='top',ha='left',linespacing=1.25))
        y=title_h
        lines.append(ax.plot([8,PAGE_PT-8],[y,y],color='#222222',lw=.8)[0])
        boxes=[]
        for pos,i in enumerate(row_ids):
            h=heights[i]; x=8
            for j,value in enumerate(wrapped[i]):
                numeric=not re.search('[A-Za-z]',plain(rows[i][j])) and i>0
                align='right' if numeric else 'left'
                xpos=x+colw[j]-6 if numeric else x+6
                font=BOLD if i==0 or '**' in rows[i][j] else FONT
                t=ax.text(xpos,y+h/2,value,fontproperties=font,ha=align,va='center',linespacing=1.2,color='#202124')
                texts.append(t);boxes.append((t,x+2,y+2,x+colw[j]-2,y+h-2))
                x+=colw[j]
            y+=h
            if pos==0:lines.append(ax.plot([8,PAGE_PT-8],[y,y],color='#555555',lw=.5)[0])
        lines.append(ax.plot([8,PAGE_PT-8],[y,y],color='#222222',lw=.8)[0])
        fig.canvas.draw();r=fig.canvas.get_renderer();outside=[];cross=[]
        clipped=[]
        for t in texts:
            bb=t.get_window_extent(r)
            if bb.x0<fig.bbox.x0-.2 or bb.y0<fig.bbox.y0-.2 or bb.x1>fig.bbox.x1+.2 or bb.y1>fig.bbox.y1+.2:
                clipped.append(t.get_text())
        for t,x0,y0,x1,y1 in boxes:
            bb=t.get_window_extent(r);coords=ax.transData.inverted().transform([[bb.x0,bb.y0],[bb.x1,bb.y1]])
            tx0,tx1=sorted(coords[:,0]);ty0,ty1=sorted(coords[:,1])
            if tx0<x0-.2 or tx1>x1+.2 or ty0<y0-.2 or ty1>y1+.2:outside.append({'text':t.get_text(),'cell':[x0,y0,x1,y1],'bbox':[tx0,ty0,tx1,ty1]})
        for t in texts:
            bb=t.get_window_extent(r)
            for line in lines:
                path=line.get_transform().transform_path(line.get_path())
                if path.intersects_bbox(bb,filled=False):cross.append(t.get_text())
        assert not outside,(stem,outside)
        assert not cross,(stem,'text intersects table rule',cross)
        assert not clipped,(stem,'text outside complete canvas',clipped)
        name=stem+(f'_part{part}' if len(groups)>1 else '')
        for extension in ('pdf','svg'):
            fig.savefig(OUT/(name+'.'+extension),facecolor='white',bbox_inches=None)
        pdf=fitz.open(OUT/(name+'.pdf'))
        assert not pdf[0].get_images(full=True),'Table PDF contains a raster'
        # Rasterize the verified vector page directly at the requested size.
        # This also keeps the PDF/PNG text placement identical across DPI changes.
        for dpi,suffix in [(1200,'_1200dpi'),(180,'_preview')]:
            pixels=pdf[0].get_pixmap(matrix=fitz.Matrix(dpi/72,dpi/72),alpha=False)
            pixels.set_dpi(dpi,dpi)
            pixels.save(OUT/(name+suffix+'.png'))
        png=Image.open(OUT/(name+'_1200dpi.png'))
        svg=(OUT/(name+'.svg')).read_text(encoding='utf-8')
        assert '<image' not in svg and '<text' in svg
        plates.append({'stem':name,'physical_mm':[180,round(height_pt/72*25.4,2)],'pixels':list(png.size),'dpi':list(png.info.get('dpi',[])),'rows':[begin,end-1],'native_vector':True,'raster_source':'direct verified vector PDF rendering, no interpolation','text_cell_boundary_violations':0,'text_rule_crossings':0,'canvas_clipped_text':0,'pdf_fonts':pdf[0].get_fonts()})
        pdf.close();plt.close(fig)
    return plates

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plt.style.use([str(STYLES/'science.mplstyle'),str(STYLES/'nature.mplstyle'),str(STYLES/'no-latex.mplstyle')])
    matplotlib.rcParams.update({'font.family':'Arial','svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42,'savefig.bbox':None,'text.usetex':False})
    manifest=[]
    for kind,filename,expected in [('main','manuscript_en_jof_v19.md',8),('supporting','supporting_information_jof_v19.md',17)]:
        source=ROOT/'manuscript'/filename;parsed=tables(source);assert len(parsed)==expected
        for idx,(caption,rows) in enumerate(parsed,1):
            stem=f'{kind}_table_'+('S' if kind=='supporting' else '')+str(idx)
            with (OUT/(stem+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:csv.writer(f).writerows([[plain(v) for v in row] for row in rows])
            plates=render_table(caption,rows,stem)
            manifest.append({'table':stem,'caption':plain(caption),'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'rows':len(rows),'columns':len(rows[0]),'cell_count':sum(map(len,rows)),'plates':plates})
    bundle=fitz.open()
    for table in manifest:
        for plate in table['plates']:
            with fitz.open(OUT/(plate['stem']+'.pdf')) as pdf:bundle.insert_pdf(pdf)
    bundle.save(OUT/'All_Tables_Vector.pdf');bundle.close()
    (OUT/'TABLE_MANIFEST.json').write_text(json.dumps({'status':'25_tables_native_vector_and_exact_source_cells','tables':manifest},indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'tables':len(manifest),'plates':sum(len(x['plates']) for x in manifest),'output':str(OUT)}))

if __name__=='__main__':main()
