from pathlib import Path
from io import StringIO
import contextlib, json, textwrap, re, html, hashlib
import pandas as pd
import numpy as np
import gzip
import nbformat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image, Preformatted, KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'informes'; OUT.mkdir(parents=True, exist_ok=True)
TMP = ROOT/'tmp/graficos'; TMP.mkdir(parents=True, exist_ok=True)
REPO = ROOT
FONT = Path(matplotlib.get_data_path())/'fonts/ttf'
for name, fname in [('Body','DejaVuSans.ttf'),('Bold','DejaVuSans-Bold.ttf'),('Mono','DejaVuSansMono.ttf')]:
    pdfmetrics.registerFont(TTFont(name, str(FONT/fname)))
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='TitleX',fontName='Bold',fontSize=23,leading=29,textColor=colors.HexColor('#13374b'),spaceAfter=16))
styles.add(ParagraphStyle(name='SubX',fontName='Bold',fontSize=13,leading=18,textColor=colors.HexColor('#15786e'),spaceBefore=14,spaceAfter=8,keepWithNext=True))
styles.add(ParagraphStyle(name='BodyX',fontName='Body',fontSize=9.4,leading=14,spaceAfter=9))
styles.add(ParagraphStyle(name='SmallX',fontName='Body',fontSize=7.7,leading=11,spaceAfter=7))
styles.add(ParagraphStyle(name='CodeX',fontName='Mono',fontSize=6.9,leading=10,spaceAfter=9,textColor=colors.HexColor('#20313e')))
styles.add(ParagraphStyle(name='LabelX',fontName='Bold',fontSize=8,leading=12,textColor=colors.HexColor('#15786e'),spaceAfter=8))

def para(s,style='BodyX'):
    return Paragraph(html.escape(s).replace('\n','<br/>'),styles[style])
def heading(s): return para(s,'SubX')
def codeblock(s):
    lines=[]
    for line in s.splitlines():
        lines.extend(textwrap.wrap(line,width=96,replace_whitespace=False,drop_whitespace=False) or [''])
    return Preformatted('\n'.join(lines),styles['CodeX'])
def title(s,course): return [para(course.upper(),'LabelX'),para(s,'TitleX'),para('Carolina Machado | 25 de octubre de 2026','SmallX')]
def table(rows,widths):
    cells=[[para(str(c),'SmallX') for c in row] for row in rows]
    t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e2f0ef')),('GRID',(0,0),(-1,-1),.35,colors.HexColor('#d0dcdf')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
    return t
def footer(c,d):
    c.setStrokeColor(colors.HexColor('#d5e1e4'));c.line(46,42,A4[0]-46,42)
    c.setFont('Body',7);c.setFillColor(colors.HexColor('#566a76'))
    c.drawString(46,29,'Carolina Machado | Coderhouse | 25/10/2026')
    c.drawRightString(A4[0]-46,29,str(d.page))
def pdf(name,story):
    p=OUT/name
    SimpleDocTemplate(str(p),pagesize=A4,leftMargin=46,rightMargin=46,topMargin=43,bottomMargin=56,title=name.removesuffix('.pdf').replace('_',' '),author='Carolina Machado').build(story,onFirstPage=footer,onLaterPages=footer)
    return p

def md(s): return nbformat.v4.new_markdown_cell(s)
def cell(s): return nbformat.v4.new_code_cell(s.strip())
cells=[md('# Campaña bancaria: estructura inicial del dataset\nCarolina Machado | Data Science 1 | Preentrega 3\n\nContinuidad: mismo Bank Marketing de UCI utilizado en la preentrega 2. Pregunta: ¿qué factores previos al contacto se asocian con la contratación de un depósito?\n\nFuente: Moro, S., Rita, P. y Cortez, P. (2014). Bank Marketing. DOI: 10.24432/C5K306. Licencia CC BY 4.0. Se usa bank-full.csv, no bank-additional-full.csv.'),
md('## 1. Carga reproducible\nEl CSV original se conserva sin modificar. El separador es punto y coma. La carpeta data incluye el diccionario original. Los resultados siguientes fueron ejecutados sobre el archivo descargado de UCI.'),
cell('''from pathlib import Path
import pandas as pd
import numpy as np
import hashlib
import gzip
pd.set_option('display.width', 110)
ruta = Path('data/bank-full.csv')
if not ruta.exists():
    # El original se conserva comprimido, sin cambiar ninguna observación.
    ruta.write_bytes(gzip.decompress(ruta.with_suffix('.csv.gz').read_bytes()))
df = pd.read_csv(ruta, sep=';')
assert df.shape == (45211, 17)
print('SHA256:', hashlib.sha256(ruta.read_bytes()).hexdigest())
print('Filas y columnas:', df.shape)
print(df.head(3).T.to_string())'''),
md('## 2. Diagnóstico estructural\nSe inspeccionan tipos y nulos técnicos. La etiqueta unknown representa información semánticamente no disponible; no equivale a un nulo de Pandas. pdays=-1 significa sin contacto previo, no un valor negativo imposible.'),
cell('''df.info()
diagnostico = pd.DataFrame({
    'tipo': df.dtypes.astype(str),
    'nulos': df.isna().sum(),
    'unknown': df.eq('unknown').sum(),
})
print(diagnostico.to_string())
print('Duplicados exactos:', df.duplicated().sum())'''),
md('## 3. Resumen numérico\nEl saldo promedio anual se expresa en euros. Se conservan los saldos negativos y extremos porque pueden corresponder a situaciones reales. La duración de la llamada se describe para auditar el origen, pero se excluye de la base predictiva por conocerse después del contacto.'),
cell('''print(df.describe().T.round(2).to_string())
print('Tasa observada de contratación (%):', round(df['y'].eq('yes').mean()*100, 2))
print('Rango de saldo (EUR):', df['balance'].min(), df['balance'].max())'''),
md('## 4. Tres filtros booleanos de validación\n1. Edad adulta y plausible entre 18 y 100 años.\n2. Al menos un contacto en la campaña actual.\n3. Recencia válida: -1 (sin contacto previo) o un número no negativo.\n\nSon controles de dominio, no una afirmación de errores inexistentes. Se registra cuántas filas excluye cada paso; un control válido puede excluir cero.'),
cell('''pasos = []
base = df.copy()
filtros = [
    ('edad entre 18 y 100', lambda x: x['age'].between(18, 100)),
    ('campaign mayor o igual a 1', lambda x: x['campaign'].ge(1)),
    ('pdays válido', lambda x: x['pdays'].eq(-1) | x['pdays'].ge(0)),
]
for nombre, filtro in filtros:
    antes = len(base)
    base = base.loc[filtro(base)].copy()
    pasos.append((nombre, antes, len(base), antes-len(base)))
print(pd.DataFrame(pasos, columns=['control','antes','despues','excluidas']).to_string(index=False))
base = base.drop_duplicates().reset_index(drop=True)
base = base.drop(columns=['duration'])
print('Base sin duration:', base.shape)
assert 'duration' not in base'''),
md('## 5. Transformaciones y agregaciones\nSe conserva unknown como categoría explícita para evitar una imputación arbitraria. La recencia se deriva sin convertir el código -1 en días reales. Las tasas son descriptivas, no estimaciones causales.'),
cell('''base['contacto_previo_grupo'] = np.select(
    [base['pdays'].eq(-1), base['pdays'].between(0, 90)],
    ['sin contacto previo', 'contacto reciente'],
    default='contacto lejano')
base['job_normalizado'] = base['job'].map(lambda x: x.strip().lower())
base['suscribe'] = base['y'].eq('yes').astype(int)
for columna in ['job_normalizado', 'education', 'contacto_previo_grupo']:
    resumen = base.groupby(columna)['suscribe'].agg(n='size', tasa='mean')
    resumen['tasa_pct'] = (resumen.pop('tasa')*100).round(2)
    print('\\nAgrupación:', columna)
    print(resumen.to_string())
base.to_csv('data/bank_saneado.csv', index=False)
print('CSV saneado guardado:', base.shape)'''),
md('## 6. Conclusión y límites\nNo se detectaron nulos técnicos ni duplicados exactos. Sí hay categorías unknown y el código pdays=-1, que se documentan y conservan. Los controles no justifican eliminar saldos extremos. Variables clave: balance, campaign, previous, poutcome, contact y sus relaciones con y. age y job se estudian descriptivamente y requieren evaluar sesgos antes de cualquier uso operativo.\n\nSe excluye duration por fuga de información. No se reconstruye una fecha completa con day y month porque falta el año por registro. En un modelo posterior habrá que precisar el momento de predicción y qué variables de campaña están disponibles en ese instante. Este análisis corresponde a un banco y un período históricos; no representa a todos los clientes ni prueba causalidad. No se entrenó ni desplegó un modelo.')]
notebook=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}})
namespace={'__name__':'__main__'}
with contextlib.chdir(REPO):
    count=0
    for c in notebook.cells:
        if c.cell_type!='code': continue
        count+=1;stream=StringIO()
        with contextlib.redirect_stdout(stream): exec(compile(c.source,f'preentrega3-celda-{count}','exec'),namespace)
        c.execution_count=count
        c.outputs=[nbformat.v4.new_output(output_type='stream',name='stdout',text=stream.getvalue())]
nbformat.write(notebook, REPO/'DataScience1_Preentrega3_CarolinaMachado.ipynb')
story=title('Estructura inicial del dataset','Data Science 1 | Preentrega 3')
story+=[para('Exportación del notebook ejecutado. Incluye código, salidas reales, diagnóstico de nulos, tipos, estadísticas y tres filtros booleanos.'),''][:-1]
for c in notebook.cells:
    if c.cell_type=='markdown':
        for line in c.source.splitlines():
            if not line: continue
            if line.startswith('# '): continue
            story.append(heading(line.lstrip('# ')) if line.startswith('## ') else para(line))
    else:
        story += [para(f'Celda ejecutada [{c.execution_count}]','LabelX'),codeblock(c.source)]
        for output in c.outputs:
            story += [para('Salida real','LabelX'),codeblock(output.text)]
pdf('Machado_Carolina_Preentrega3_Dataset.pdf',story)

base = namespace['base']
# El EDA parte del CSV saneado que produjo el notebook anterior.
df = pd.read_csv(REPO / 'data/bank_saneado.csv')
assert len(df) == len(base)
assert 'duration' not in df.columns
sns.set_theme(style='whitegrid',font='DejaVu Sans',palette=['#197e78','#d88d36'])
charts=[]
def savechart(n,title_,note,draw):
    fig,ax=plt.subplots(figsize=(8,4.4));draw(ax);ax.set_title(title_,loc='left',fontsize=12,fontweight='bold',pad=14)
    fig.tight_layout();p=TMP/f'ds4_{n}.png';fig.savefig(p,dpi=165);plt.close(fig);charts.append((title_,note,p))
savechart(1,'Distribución de edades',f"La mediana de edad es {df.age.median():.0f} años; el 50 % central se encuentra entre {df.age.quantile(.25):.0f} y {df.age.quantile(.75):.0f}. La presencia de edades altas requiere conservar la heterogeneidad de la muestra, sin tratarlas automáticamente como errores.",lambda ax:sns.histplot(data=df,x='age',bins=30,kde=True,ax=ax,color='#197e78').set(xlabel='Edad (años)',ylabel='Cantidad de contactos'))
df['saldo_log_firmado']=np.sign(df.balance)*np.log1p(np.abs(df.balance))
savechart(2,'Distribución del saldo con escala transformada',f"El saldo original tiene mediana {df.balance.median():,.0f} EUR y extremos de {df.balance.min():,} a {df.balance.max():,} EUR. Se visualiza signo(saldo) × ln(1 + |saldo|) para hacer visibles saldos negativos, cero y positivos; esta transformación es solo visual y no elimina observaciones.",lambda ax:sns.histplot(data=df,x='saldo_log_firmado',bins=45,ax=ax,color='#197e78').set(xlabel='signo(saldo) × ln(1 + |saldo en EUR|)',ylabel='Cantidad de contactos'))
savechart(3,'Edades según resultado de la campaña','Los rangos de edad de ambos resultados se superponen ampliamente. Las diferencias de mediana o dispersión describen la muestra; no permiten atribuir una mayor contratación a la edad ni justifican decisiones automáticas sobre personas.',lambda ax:sns.boxplot(data=df,x='y',y='age',order=['no','yes'],ax=ax,color='#80b8b2').set(xlabel='Contrató depósito (no / yes)',ylabel='Edad (años)'))
muestra=df.sample(n=2000,random_state=42)
savechart(4,'Edad y saldo: relación y dispersión','Muestra aleatoria de 2.000 contactos, con semilla 42, tomada solo para reducir la superposición gráfica. El saldo usa escala symlog para mostrar negativos y extremos. No aparece una separación simple y visual entre quienes contrataron y quienes no; una asociación visual no es causalidad.',lambda ax:(sns.scatterplot(data=muestra,x='age',y='balance',hue='y',hue_order=['no','yes'],alpha=.4,s=19,ax=ax),ax.set_yscale('symlog',linthresh=100),ax.set(xlabel='Edad (años)',ylabel='Saldo promedio anual (EUR, symlog)'),ax.legend(title='Contrató')))
tasas=df.assign(suscribe=df.y.eq('yes')).groupby('job').suscribe.agg(['mean','size']).sort_values('mean')
def bar(ax):
    ax.barh(tasas.index,tasas['mean']*100,color='#197e78');ax.set(xlabel='Contratación observada (%)',ylabel='Ocupación declarada')
    for y,(_,r) in enumerate(tasas.iterrows()):ax.text(r['mean']*100+.35,y,f"n={r['size']:.0f}",va='center',fontsize=8)
    ax.set_xlim(0,tasas['mean'].max()*100+7)
savechart(5,'Contratación por ocupación y tamaño del grupo','Cada barra muestra la tasa observada y el n de su categoría. Los grupos difieren en composición y tamaño; student y retired pueden mostrar tasas distintas sin que la ocupación explique por sí sola el resultado. Deben revisarse incertidumbre, sesgo y momento de disponibilidad antes de modelar.',bar)
story=title('EDA visual de campañas bancarias','Data Science 1 | Preentrega 4')
story += [para(f"Bank Marketing (UCI), bank-full.csv: {len(df):,} contactos y 17 columnas originales. La contratación observada es {df.y.eq('yes').mean()*100:.2f} %. Se continúa el dataset saneado de la preentrega 3, sin eliminar valores extremos válidos. Los gráficos usan Matplotlib y Seaborn; los dos primeros son univariados y los tres siguientes comparan variables."),para('Fuente: Moro, Rita y Cortez (2014). UCI Bank Marketing. DOI 10.24432/C5K306. CC BY 4.0.','SmallX')]
for i,(name,note,p) in enumerate(charts):
    if i:story.append(PageBreak())
    story += [heading(f'{i+1}. {name}'),Image(str(p),width=502,height=276),Spacer(1,10),para(note)]
    if i==4:story += [heading('Conclusión y siguiente paso'),para('La muestra es heterogénea y la clase positiva es minoritaria. El análisis siguiente debe controlar la fuga de información de duration, separar entrenamiento y evaluación antes de ajustar transformaciones y comparar modelos con métricas apropiadas al desbalance. Los resultados describen este banco histórico; no se generalizan automáticamente a otra población. No se construyó un modelo ni se tomaron decisiones sobre clientes.')]
pdf('Machado_Carolina_Preentrega4_EDA_Visual.pdf',story)


print('Informes generados en', OUT)
