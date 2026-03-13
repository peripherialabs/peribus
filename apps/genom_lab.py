"""
GenomeLab — Genomics & Bioinformatics Workbench for Rio
═══════════════════════════════════════════════════════════════════

ARCHITECTURE: This file is the complete runtime. It registers a `genome`
singleton into the namespace. The LLM sees this source as context and
injects short code snippets to drive it.

EXAMPLES (what the LLM would output):
    genome.load_vcf("/path/to/variants.vcf")        # load VCF file
    genome.load_fasta("/path/to/sequence.fa")        # load FASTA reference
    genome.load_bed("/path/to/regions.bed")          # load BED intervals
    genome.load_gff("/path/to/genes.gff3")           # load gene annotations
    genome.load_demo("brca")                         # load demo dataset

    ## Navigation & display:
    genome.goto("chr17:41196312-41277500")           # jump to region (BRCA1)
    genome.goto("chr7", 117120017, 117308719)        # CFTR locus
    genome.zoom_in()                                 # 2x zoom into center
    genome.zoom_out()                                # 2x zoom out
    genome.set_chromosome("chr1")                    # switch chromosome view
    genome.ideogram()                                # show full karyotype

    ## Variant analysis:
    genome.filter_variants(chrom="chr17", min_qual=30)
    genome.filter_variants(gene="BRCA1")
    genome.filter_variants(impact="HIGH")
    genome.variant_summary()                         # counts by type, chrom
    genome.annotate()                                # functional annotation
    genome.overlay_manhattan()                       # Manhattan plot
    genome.overlay_frequency()                       # allele frequency histogram
    genome.overlay_qual()                            # quality score distribution

    ## Sequence operations:
    genome.show_sequence("chr17", 41197694, 41197819)  # show nucleotide seq
    genome.gc_content("chr17", 41196312, 41277500)     # GC% in region
    genome.find_motif("GAATTC")                        # find EcoRI sites
    genome.translate("chr17", 41197694, 41197819)      # 6-frame translation

    ## Gene models:
    genome.show_genes()                              # overlay gene models in view
    genome.gene_info("BRCA1")                        # lookup gene details
    genome.overlay_coverage()                        # show read depth plot

    ## Export:
    genome.export_bed("/tmp/filtered.bed")           # export filtered regions
    genome.export_fasta("/tmp/region.fa", "chr17", 100, 200)
    genome.export_vcf("/tmp/filtered.vcf")           # export filtered variants
    genome.stats()                                   # genome-wide statistics

    ## After loading, genome.info contains:
    genome.info['filename']        # loaded file name
    genome.info['n_variants']      # total variant count
    genome.info['n_sequences']     # number of sequences/chromosomes
    genome.info['total_length']    # total genome length in bp
    genome.info['variant_types']   # dict of variant type counts
    genome.info['chromosomes']     # list of chromosome names
    genome.info['genes']           # list of gene records
    genome.info['assembly']        # genome assembly name if detected

VIEWER API (lower level, when LLM needs custom rendering):
    genome.viewer.set_region(chrom, start, end)
    genome.viewer.add_track(name, track_data)
    genome.viewer.remove_track(name)
    genome.viewer.add_overlay(name, fn)            # fn(painter, w, h) for QPainter
    genome.viewer.remove_overlay(name)
    genome.viewer.cam_x, genome.viewer.cam_y       # pan position
    genome.viewer.zoom                              # zoom level
    genome.viewer.screenshot(path)                  # save PNG

NAMESPACE: After this file runs, these are available:
    genome      — GenomeLab singleton (main API)
    viewer      — alias for genome.viewer (the GL viewer widget)
    gen         — alias for genome (short form)
    CHROMOSOMES — human chromosome data dict
    All PySide6/Qt, numpy, moderngl, glm from parser namespace
"""

import math
import os
import re
import gzip
import threading
import numpy as np
from collections import OrderedDict, defaultdict

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QSlider, QComboBox, QCheckBox, QTabWidget, QTextEdit,
    QScrollArea, QListWidget, QLineEdit, QGraphicsItem,
    QGraphicsDropShadowEffect, QSizePolicy, QProgressBar
)
from PySide6.QtCore import Qt, QTimer, Signal, QObject
from PySide6.QtGui import (
    QPainter, QColor, QFont, QPen, QBrush, QImage, QLinearGradient,
    QPainterPath
)

import moderngl
import glm
import json

# ═══════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════

def _hex(h):
    return ((h >> 16) & 0xFF) / 255.0, ((h >> 8) & 0xFF) / 255.0, (h & 0xFF) / 255.0

def _qhex(h):
    return QColor((h >> 16) & 0xFF, (h >> 8) & 0xFF, h & 0xFF)

# Human chromosome sizes (GRCh38) and cytogenetic band colors
CHROMOSOMES = OrderedDict([
    ("chr1",  {"length": 248956422, "color": 0xE8544E, "centromere": (121700000, 125100000)}),
    ("chr2",  {"length": 242193529, "color": 0xE8894E, "centromere": (91800000, 96000000)}),
    ("chr3",  {"length": 198295559, "color": 0xE8C44E, "centromere": (87800000, 94000000)}),
    ("chr4",  {"length": 190214555, "color": 0xC4E84E, "centromere": (48200000, 51800000)}),
    ("chr5",  {"length": 181538259, "color": 0x89E84E, "centromere": (46100000, 51400000)}),
    ("chr6",  {"length": 170805979, "color": 0x4EE854, "centromere": (58500000, 63300000)}),
    ("chr7",  {"length": 159345973, "color": 0x4EE889, "centromere": (58100000, 62100000)}),
    ("chr8",  {"length": 145138636, "color": 0x4EE8C4, "centromere": (43600000, 47200000)}),
    ("chr9",  {"length": 138394717, "color": 0x4EC4E8, "centromere": (42200000, 45500000)}),
    ("chr10", {"length": 133797422, "color": 0x4E89E8, "centromere": (38000000, 41600000)}),
    ("chr11", {"length": 135086622, "color": 0x4E54E8, "centromere": (51000000, 55800000)}),
    ("chr12", {"length": 133275309, "color": 0x894EE8, "centromere": (33200000, 37800000)}),
    ("chr13", {"length": 114364328, "color": 0xC44EE8, "centromere": (16500000, 18900000)}),
    ("chr14", {"length": 107043718, "color": 0xE84EC4, "centromere": (16100000, 18200000)}),
    ("chr15", {"length": 101991189, "color": 0xE84E89, "centromere": (17500000, 20500000)}),
    ("chr16", {"length": 90338345,  "color": 0xFF6B6B, "centromere": (35300000, 38400000)}),
    ("chr17", {"length": 83257441,  "color": 0xFFA07A, "centromere": (22700000, 27400000)}),
    ("chr18", {"length": 80373285,  "color": 0xFFD700, "centromere": (15400000, 21500000)}),
    ("chr19", {"length": 58617616,  "color": 0x98FB98, "centromere": (24200000, 28100000)}),
    ("chr20", {"length": 64444167,  "color": 0x87CEEB, "centromere": (25700000, 30400000)}),
    ("chr21", {"length": 46709983,  "color": 0xDDA0DD, "centromere": (10900000, 13000000)}),
    ("chr22", {"length": 50818468,  "color": 0xBC8F8F, "centromere": (13700000, 17400000)}),
    ("chrX",  {"length": 156040895, "color": 0xFF69B4, "centromere": (58100000, 63800000)}),
    ("chrY",  {"length": 57227415,  "color": 0x6495ED, "centromere": (10300000, 10600000)}),
])

TOTAL_GENOME = sum(c["length"] for c in CHROMOSOMES.values())

# Genetic code
CODON_TABLE = {
    'TTT':'F','TTC':'F','TTA':'L','TTG':'L','CTT':'L','CTC':'L','CTA':'L','CTG':'L',
    'ATT':'I','ATC':'I','ATA':'I','ATG':'M','GTT':'V','GTC':'V','GTA':'V','GTG':'V',
    'TCT':'S','TCC':'S','TCA':'S','TCG':'S','CCT':'P','CCC':'P','CCA':'P','CCG':'P',
    'ACT':'T','ACC':'T','ACA':'T','ACG':'T','GCT':'A','GCC':'A','GCA':'A','GCG':'A',
    'TAT':'Y','TAC':'Y','TAA':'*','TAG':'*','CAT':'H','CAC':'H','CAA':'Q','CAG':'Q',
    'AAT':'N','AAC':'N','AAA':'K','AAG':'K','GAT':'D','GAC':'D','GAA':'E','GAG':'E',
    'TGT':'C','TGC':'C','TGA':'*','TGG':'W','CGT':'R','CGC':'R','CGA':'R','CGG':'R',
    'AGT':'S','AGC':'S','AGA':'R','AGG':'R','GGT':'G','GGC':'G','GGA':'G','GGG':'G',
}

COMPLEMENT = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N',
              'a': 't', 't': 'a', 'g': 'c', 'c': 'g', 'n': 'n'}

# Nucleotide colors
NUC_COLORS = {
    'A': 0x4CAF50, 'T': 0xF44336, 'G': 0xFFC107, 'C': 0x2196F3,
    'a': 0x4CAF50, 't': 0xF44336, 'g': 0xFFC107, 'c': 0x2196F3,
    'N': 0x9E9E9E, 'n': 0x9E9E9E,
}

# Impact severity colors
IMPACT_COLORS = {
    'HIGH':     0xD32F2F,
    'MODERATE': 0xF57C00,
    'LOW':      0xFBC02D,
    'MODIFIER': 0x9E9E9E,
}

# Variant type colors
VARTYPE_COLORS = {
    'SNV':    0x5C6BC0,
    'INS':    0x66BB6A,
    'DEL':    0xEF5350,
    'MNV':    0xFFA726,
    'OTHER':  0x9E9E9E,
}

# ═══════════════════════════════════════════════════════════════
#  DEMO DATASETS  (synthetic but realistic)
# ═══════════════════════════════════════════════════════════════

def _make_demo_brca():
    """Generate a demo VCF-like dataset centered on BRCA1/BRCA2."""
    np.random.seed(42)
    variants = []
    genes = [
        {"name": "BRCA1", "chrom": "chr17", "start": 41196312, "end": 41277500,
         "strand": "-", "exons": [(41196312,41197819),(41199660,41199720),(41201138,41201211),
                                   (41203080,41203134),(41209068,41209152),(41215349,41215390),
                                   (41219624,41219712),(41222944,41223255),(41226348,41228631),
                                   (41234421,41234592),(41242960,41243049),(41243452,41246877),
                                   (41247862,41247939),(41249261,41249306),(41251792,41251897),
                                   (41254170,41254436),(41256139,41256278),(41258473,41258550),
                                   (41267743,41267796),(41276034,41277287)],
         "biotype": "protein_coding", "description": "BRCA1 DNA repair associated"},
        {"name": "BRCA2", "chrom": "chr13", "start": 32315480, "end": 32400268,
         "strand": "+", "exons": [(32315480,32315667),(32316422,32316527),(32319077,32319325),
                                   (32325076,32325184),(32326101,32326282),(32326499,32326613),
                                   (32329443,32329492),(32330920,32331030),(32332272,32333387),
                                   (32336265,32341196),(32344558,32344653),(32346827,32346896),
                                   (32354861,32355288),(32356428,32370557),(32370956,32371100),
                                   (32376670,32376791),(32379317,32379515),(32379750,32379913),
                                   (32380007,32380145),(32394689,32394933),(32396898,32397044),
                                   (32398162,32398770),(32399671,32400268)],
         "biotype": "protein_coding", "description": "BRCA2 DNA repair associated"},
        {"name": "TP53", "chrom": "chr17", "start": 7668402, "end": 7687550,
         "strand": "-", "exons": [(7668402,7669690),(7670609,7670715),(7673535,7673608),
                                   (7673701,7673837),(7674181,7674290),(7674859,7674971),
                                   (7675053,7675236),(7675994,7676272),(7676381,7676403),
                                   (7676521,7676622),(7687377,7687550)],
         "biotype": "protein_coding", "description": "Tumor protein p53"},
        {"name": "EGFR", "chrom": "chr7", "start": 55019017, "end": 55211628,
         "strand": "+", "exons": [(55019017,55019365),(55142285,55142439),(55143305,55143488),
                                   (55146607,55146742),(55151294,55151362),(55152546,55152665),
                                   (55154010,55154152),(55155830,55155946),(55156533,55156680),
                                   (55157663,55157753),(55160139,55160338),(55161499,55161631),
                                   (55163733,55163823),(55165280,55165411),(55168517,55168632),
                                   (55170304,55170470),(55172975,55173124),(55173919,55174043),
                                   (55177416,55177548),(55181293,55181478),(55191719,55191874),
                                   (55198717,55198863),(55200320,55200416),(55201187,55201317),
                                   (55201708,55201860),(55204709,55204869),(55205406,55211628)],
         "biotype": "protein_coding", "description": "Epidermal growth factor receptor"},
    ]
    # Generate ~300 variants spread across chromosomes with concentration near genes
    chroms = list(CHROMOSOMES.keys())[:22]  # autosomes
    for _ in range(200):
        ch = np.random.choice(chroms)
        pos = int(np.random.uniform(1, CHROMOSOMES[ch]['length']))
        ref = np.random.choice(['A','C','G','T'])
        alt = np.random.choice([b for b in ['A','C','G','T'] if b != ref])
        qual = float(np.random.exponential(40) + 10)
        af = round(float(np.random.beta(2, 5)), 4)
        vtype = 'SNV'
        impact = np.random.choice(['MODIFIER','LOW','MODERATE'], p=[0.6,0.25,0.15])
        variants.append({"chrom": ch, "pos": pos, "id": ".", "ref": ref, "alt": alt,
                         "qual": round(qual, 1), "filter": "PASS" if qual > 20 else "LowQual",
                         "af": af, "type": vtype, "impact": impact, "gene": "."})
    # Cluster variants near known genes
    for gene in genes:
        n_near = np.random.randint(8, 25)
        for _ in range(n_near):
            pos = int(np.random.uniform(gene['start'] - 5000, gene['end'] + 5000))
            ref = np.random.choice(['A','C','G','T'])
            # Mix of SNVs, insertions, deletions
            r = np.random.random()
            if r < 0.7:
                alt = np.random.choice([b for b in ['A','C','G','T'] if b != ref])
                vtype = 'SNV'
            elif r < 0.85:
                alt = ref + ''.join(np.random.choice(['A','C','G','T']) for _ in range(np.random.randint(1,5)))
                vtype = 'INS'
            else:
                ref = ''.join(np.random.choice(['A','C','G','T']) for _ in range(np.random.randint(2,6)))
                alt = ref[0]
                vtype = 'DEL'
            qual = float(np.random.exponential(60) + 20)
            af = round(float(np.random.beta(2, 8)), 4)
            # Higher impact near exons
            in_exon = any(s <= pos <= e for s, e in gene.get('exons', []))
            if in_exon:
                impact = np.random.choice(['HIGH','MODERATE','LOW'], p=[0.3,0.5,0.2])
            else:
                impact = np.random.choice(['MODIFIER','LOW','MODERATE'], p=[0.5,0.35,0.15])
            variants.append({"chrom": gene['chrom'], "pos": pos, "id": ".", "ref": ref,
                             "alt": alt, "qual": round(qual, 1),
                             "filter": "PASS" if qual > 20 else "LowQual",
                             "af": af, "type": vtype, "impact": impact, "gene": gene['name']})
    # Sort
    chrom_order = {c: i for i, c in enumerate(CHROMOSOMES.keys())}
    variants.sort(key=lambda v: (chrom_order.get(v['chrom'], 99), v['pos']))
    # Generate some fake coverage data as a sparse array
    coverage = {}
    for gene in genes:
        ch = gene['chrom']
        if ch not in coverage:
            coverage[ch] = []
        step = max(1, (gene['end'] - gene['start']) // 200)
        for p in range(gene['start'] - 2000, gene['end'] + 2000, step):
            # Higher coverage in exons
            in_exon = any(s <= p <= e for s, e in gene.get('exons', []))
            base_cov = np.random.poisson(60 if in_exon else 30)
            coverage[ch].append((p, int(base_cov)))
    # Generate a small reference sequence snippet for BRCA1 region
    seq_snippet = {
        "chr17": {
            "start": 41196312,
            "seq": ''.join(np.random.choice(['A','C','G','T'], p=[0.29,0.21,0.21,0.29])
                          for _ in range(2000))
        }
    }
    return {"variants": variants, "genes": genes, "coverage": coverage,
            "sequences": seq_snippet, "assembly": "GRCh38"}

DEMOS = {
    "brca": _make_demo_brca,
}

# ═══════════════════════════════════════════════════════════════
#  PARSERS
# ═══════════════════════════════════════════════════════════════

def _open_maybe_gz(path):
    """Open a file, auto-detecting gzip."""
    path = os.path.expanduser(path)
    if path.endswith('.gz'):
        return gzip.open(path, 'rt')
    return open(path, 'r')

def parse_vcf(path):
    """Parse VCF file → list of variant dicts."""
    variants = []
    header_lines = []
    samples = []
    with _open_maybe_gz(path) as f:
        for line in f:
            if line.startswith('##'):
                header_lines.append(line.strip())
                continue
            if line.startswith('#CHROM'):
                parts = line.strip().split('\t')
                if len(parts) > 9:
                    samples = parts[9:]
                continue
            parts = line.strip().split('\t')
            if len(parts) < 8:
                continue
            chrom, pos, vid, ref, alt, qual, filt, info_str = parts[:8]
            # Parse INFO field
            info = {}
            for kv in info_str.split(';'):
                if '=' in kv:
                    k, v = kv.split('=', 1)
                    info[k] = v
                else:
                    info[kv] = True
            # Determine variant type
            if len(ref) == 1 and len(alt) == 1:
                vtype = 'SNV'
            elif len(ref) < len(alt):
                vtype = 'INS'
            elif len(ref) > len(alt):
                vtype = 'DEL'
            elif len(ref) == len(alt) and len(ref) > 1:
                vtype = 'MNV'
            else:
                vtype = 'OTHER'
            try:
                qual_f = float(qual) if qual != '.' else 0.0
            except:
                qual_f = 0.0
            # Extract AF
            af = 0.0
            if 'AF' in info:
                try:
                    af = float(info['AF'].split(',')[0])
                except:
                    pass
            # Extract impact/gene from ANN or CSQ if present
            impact = info.get('IMPACT', 'MODIFIER')
            gene = info.get('GENE', info.get('SYMBOL', '.'))
            if 'ANN' in info:
                ann_parts = str(info['ANN']).split('|')
                if len(ann_parts) > 3:
                    gene = ann_parts[3] if ann_parts[3] else gene
                if len(ann_parts) > 2:
                    impact = ann_parts[2] if ann_parts[2] in IMPACT_COLORS else impact
            variants.append({
                "chrom": chrom, "pos": int(pos), "id": vid, "ref": ref, "alt": alt,
                "qual": qual_f, "filter": filt, "af": af, "type": vtype,
                "impact": impact, "gene": gene, "info": info
            })
    return variants, header_lines, samples

def parse_fasta(path):
    """Parse FASTA → dict of {name: sequence_string}."""
    sequences = {}
    current_name = None
    current_seq = []
    with _open_maybe_gz(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_name is not None:
                    sequences[current_name] = ''.join(current_seq)
                current_name = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
    if current_name is not None:
        sequences[current_name] = ''.join(current_seq)
    return sequences

def parse_bed(path):
    """Parse BED file → list of region dicts."""
    regions = []
    with _open_maybe_gz(path) as f:
        for line in f:
            if line.startswith('#') or line.startswith('track') or line.startswith('browser'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            region = {"chrom": parts[0], "start": int(parts[1]), "end": int(parts[2])}
            if len(parts) > 3: region["name"] = parts[3]
            if len(parts) > 4:
                try: region["score"] = float(parts[4])
                except: pass
            if len(parts) > 5: region["strand"] = parts[5]
            regions.append(region)
    return regions

def parse_gff(path):
    """Parse GFF3/GTF → list of gene/feature dicts."""
    genes = {}
    features = []
    with _open_maybe_gz(path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 9:
                continue
            chrom, source, ftype, start, end, score, strand, phase, attrs_str = parts
            attrs = {}
            for kv in attrs_str.split(';'):
                kv = kv.strip()
                if '=' in kv:
                    k, v = kv.split('=', 1)
                    attrs[k] = v
                elif ' "' in kv:
                    m = re.match(r'(\S+)\s+"([^"]*)"', kv)
                    if m: attrs[m.group(1)] = m.group(2)
            feat = {"chrom": chrom, "source": source, "type": ftype,
                    "start": int(start), "end": int(end), "strand": strand, "attrs": attrs}
            features.append(feat)
            # Collect gene-level records
            gene_name = attrs.get('gene_name', attrs.get('Name', attrs.get('gene_id', '')))
            if ftype == 'gene' and gene_name:
                genes[gene_name] = {
                    "name": gene_name, "chrom": chrom, "start": int(start), "end": int(end),
                    "strand": strand, "biotype": attrs.get('gene_biotype', attrs.get('gene_type', '')),
                    "description": attrs.get('description', ''), "exons": []
                }
            elif ftype in ('exon', 'CDS') and gene_name and gene_name in genes:
                genes[gene_name]["exons"].append((int(start), int(end)))
    # Sort exons
    for g in genes.values():
        g["exons"].sort()
    return list(genes.values()), features

# ═══════════════════════════════════════════════════════════════
#  SHADERS
# ═══════════════════════════════════════════════════════════════

_VERT_2D = """
#version 330
uniform mat4 mvp;
in vec2 in_position;
in vec3 in_color;
out vec3 v_color;
void main() {
    gl_Position = mvp * vec4(in_position, 0.0, 1.0);
    v_color = in_color;
}"""

_FRAG_2D = """
#version 330
in vec3 v_color;
out vec4 frag;
void main() {
    frag = vec4(v_color, 1.0);
}"""

_FRAG_2D_ALPHA = """
#version 330
uniform float alpha;
in vec3 v_color;
out vec4 frag;
void main() {
    frag = vec4(v_color, alpha);
}"""

# ═══════════════════════════════════════════════════════════════
#  GEOMETRY HELPERS
# ═══════════════════════════════════════════════════════════════

def _make_rect_2d(x, y, w, h, color_hex):
    """Create a 2D filled rectangle as 2 triangles: pos(2)+color(3) per vertex."""
    r, g, b = _hex(color_hex)
    return [
        x, y, r, g, b,        x+w, y, r, g, b,      x+w, y+h, r, g, b,
        x, y, r, g, b,        x+w, y+h, r, g, b,    x, y+h, r, g, b,
    ]

def _make_rounded_rect_2d(x, y, w, h, radius, color_hex, segs=6):
    """Create a 2D rounded rectangle."""
    r, g, b = _hex(color_hex)
    verts = []
    cx, cy = x + w / 2, y + h / 2
    # Build as triangle fan from center
    corners = [
        (x + radius, y + radius, math.pi, 1.5 * math.pi),
        (x + w - radius, y + radius, 1.5 * math.pi, 2 * math.pi),
        (x + w - radius, y + h - radius, 0, 0.5 * math.pi),
        (x + radius, y + h - radius, 0.5 * math.pi, math.pi),
    ]
    points = []
    for (ccx, ccy, a_start, a_end) in corners:
        for i in range(segs + 1):
            a = a_start + (a_end - a_start) * i / segs
            points.append((ccx + radius * math.cos(a), ccy + radius * math.sin(a)))
    # Triangle fan
    for i in range(len(points)):
        p1 = points[i]
        p2 = points[(i + 1) % len(points)]
        verts.extend([cx, cy, r, g, b, p1[0], p1[1], r, g, b, p2[0], p2[1], r, g, b])
    return verts

# ═══════════════════════════════════════════════════════════════
#  GENOME VIEWER WIDGET (2D chromosome/track viewer)
# ═══════════════════════════════════════════════════════════════

class GenomeViewer(QWidget):
    """2D genomic region viewer — offscreen FBO → QImage → QPainter.
    Displays chromosome ideograms, variant tracks, gene models, coverage."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(500, 350)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        # View state
        self.chrom = "chr17"
        self.view_start = 41196312
        self.view_end = 41277500
        self.zoom = 1.0
        self._dragging = False
        self._lmx = 0

        # Data
        self.variants = []
        self.genes = []
        self.coverage = {}       # {chrom: [(pos, depth), ...]}
        self.sequences = {}      # {chrom: {"start": int, "seq": str}}
        self.bed_regions = []
        self.tracks = OrderedDict()  # name → track rendering data

        # Display modes
        self.show_ideogram = True
        self.show_variants = True
        self.show_genes = True
        self.show_coverage = False
        self.show_sequence = False
        self.color_by = 'type'   # 'type', 'impact', 'qual'

        # GL
        self._gl_ready = False
        self.ctx = None
        self.fbo = None
        self._fbo_w = 0
        self._fbo_h = 0
        self._frame = None

        # Overlays: dict of name → fn(painter, w, h)
        self._overlays = OrderedDict()

        # Hover info
        self._hover_variant = None

        # Timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.setInterval(33)

    def _ensure_gl(self):
        if self._gl_ready:
            return
        self._gl_ready = True
        self.ctx = moderngl.create_context(standalone=True)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self.prog = self.ctx.program(vertex_shader=_VERT_2D, fragment_shader=_FRAG_2D)
        self.prog_alpha = self.ctx.program(vertex_shader=_VERT_2D, fragment_shader=_FRAG_2D_ALPHA)
        self._resize_fbo(max(self.width(), 400), max(self.height(), 300))
        self.timer.start()

    def _resize_fbo(self, w, h):
        if w == self._fbo_w and h == self._fbo_h and self.fbo:
            return
        if self.fbo:
            self.fbo.release()
        self._fbo_w = w
        self._fbo_h = h
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((w, h), 4)],
            depth_attachment=self.ctx.depth_renderbuffer((w, h)))

    def set_region(self, chrom, start, end):
        self.chrom = chrom
        self.view_start = max(0, int(start))
        self.view_end = max(self.view_start + 100, int(end))

    def add_overlay(self, name, fn):
        self._overlays[name] = fn

    def remove_overlay(self, name):
        self._overlays.pop(name, None)

    def screenshot(self, path):
        if self._frame:
            self._frame.save(path)

    def _bp_to_x(self, pos, w, margin_l=60, margin_r=20):
        """Convert a base-pair position to x pixel coordinate."""
        span = max(1, self.view_end - self.view_start)
        draw_w = w - margin_l - margin_r
        return margin_l + (pos - self.view_start) / span * draw_w

    def _x_to_bp(self, x, w, margin_l=60, margin_r=20):
        """Convert x pixel to base-pair position."""
        span = max(1, self.view_end - self.view_start)
        draw_w = w - margin_l - margin_r
        return self.view_start + (x - margin_l) / draw_w * span

    def _tick(self):
        self._render()
        self.update()

    def _render(self):
        if not self._gl_ready:
            return
        w, h = max(self.width(), 400), max(self.height(), 300)
        self._resize_fbo(w, h)
        self.fbo.use()
        self.ctx.viewport = (0, 0, w, h)
        self.ctx.clear(0, 0, 0, 0)
        # GL rendering is minimal — we draw most via QPainter for text quality
        raw = self.fbo.color_attachments[0].read()
        self._frame = QImage(raw, w, h, w * 4, QImage.Format_RGBA8888).mirrored(False, True)

    def paintEvent(self, event):
        self._ensure_gl()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        # Background
        p.fillRect(0, 0, w, h, QColor(252, 253, 255, 240))

        margin_l, margin_r, top = 60, 20, 10
        draw_w = w - margin_l - margin_r
        span = max(1, self.view_end - self.view_start)

        # ── Region header ──
        p.setFont(QFont("Consolas", 10, QFont.Bold))
        p.setPen(QColor(40, 50, 70))
        region_str = f"{self.chrom}:{self.view_start:,}-{self.view_end:,}"
        span_str = self._format_bp(span)
        p.drawText(margin_l, top, draw_w, 20, Qt.AlignLeft | Qt.AlignVCenter,
                   f"{region_str}  ({span_str})")

        y_cursor = top + 28

        # ── Chromosome ideogram bar ──
        if self.show_ideogram and self.chrom in CHROMOSOMES:
            y_cursor = self._draw_ideogram(p, w, h, y_cursor, margin_l, margin_r)
            y_cursor += 12

        # ── Coordinate ruler ──
        y_cursor = self._draw_ruler(p, w, y_cursor, margin_l, margin_r)
        y_cursor += 8

        # ── Gene track ──
        if self.show_genes and self.genes:
            y_cursor = self._draw_gene_track(p, w, y_cursor, margin_l, margin_r)
            y_cursor += 8

        # ── Variant track ──
        if self.show_variants and self.variants:
            y_cursor = self._draw_variant_track(p, w, y_cursor, margin_l, margin_r)
            y_cursor += 8

        # ── Coverage track ──
        if self.show_coverage and self.chrom in self.coverage:
            y_cursor = self._draw_coverage_track(p, w, y_cursor, margin_l, margin_r)
            y_cursor += 8

        # ── Sequence track (only at high zoom) ──
        if self.show_sequence and span < 500 and self.chrom in self.sequences:
            y_cursor = self._draw_sequence_track(p, w, y_cursor, margin_l, margin_r)

        # ── Custom overlays ──
        for name, fn in self._overlays.items():
            try:
                fn(p, w, h)
            except:
                pass

        # ── Legend ──
        self._draw_legend(p, w, h)

        p.end()

    def _format_bp(self, bp):
        if bp >= 1e9: return f"{bp/1e9:.1f} Gb"
        if bp >= 1e6: return f"{bp/1e6:.1f} Mb"
        if bp >= 1e3: return f"{bp/1e3:.1f} kb"
        return f"{bp} bp"

    def _draw_ideogram(self, painter, w, h, y, ml, mr):
        """Draw chromosome ideogram with centromere."""
        ch_info = CHROMOSOMES[self.chrom]
        ch_len = ch_info['length']
        draw_w = w - ml - mr
        bar_h = 14

        # Full chromosome bar
        painter.setPen(QPen(QColor(180, 190, 210), 1))
        painter.setBrush(QColor(230, 235, 245))
        painter.drawRoundedRect(ml, y, draw_w, bar_h, 5, 5)

        # Centromere
        cs, ce = ch_info['centromere']
        cx1 = ml + cs / ch_len * draw_w
        cx2 = ml + ce / ch_len * draw_w
        painter.setBrush(QColor(200, 160, 160, 180))
        painter.setPen(Qt.NoPen)
        painter.drawRect(int(cx1), y, int(cx2 - cx1), bar_h)

        # Current view indicator
        vx1 = ml + self.view_start / ch_len * draw_w
        vx2 = ml + self.view_end / ch_len * draw_w
        painter.setBrush(QColor(70, 130, 200, 100))
        painter.setPen(QPen(QColor(50, 100, 180), 1.5))
        painter.drawRect(int(vx1), y - 2, max(2, int(vx2 - vx1)), bar_h + 4)

        # Label
        painter.setFont(QFont("Consolas", 8))
        painter.setPen(QColor(100, 110, 130))
        painter.drawText(4, y, ml - 8, bar_h, Qt.AlignRight | Qt.AlignVCenter, self.chrom)

        return y + bar_h

    def _draw_ruler(self, painter, w, y, ml, mr):
        """Draw genomic coordinate ruler with tick marks."""
        draw_w = w - ml - mr
        span = max(1, self.view_end - self.view_start)

        # Choose tick spacing
        raw_step = span / 8
        magnitude = 10 ** int(math.log10(max(1, raw_step)))
        nice_steps = [1, 2, 5, 10, 20, 50, 100]
        step = magnitude
        for ns in nice_steps:
            if ns * magnitude >= raw_step:
                step = ns * magnitude
                break

        painter.setFont(QFont("Consolas", 7))
        painter.setPen(QPen(QColor(180, 190, 210), 1))

        # Draw ticks
        tick_start = int(self.view_start / step) * step
        for tick_bp in range(int(tick_start), int(self.view_end) + 1, max(1, int(step))):
            if tick_bp < self.view_start:
                continue
            tx = self._bp_to_x(tick_bp, w, ml, mr)
            painter.drawLine(int(tx), y, int(tx), y + 6)
            painter.setPen(QColor(130, 140, 160))
            label = self._format_bp(tick_bp) if tick_bp > 0 else "0"
            painter.drawText(int(tx) - 30, y + 7, 60, 12, Qt.AlignCenter, label)
            painter.setPen(QPen(QColor(180, 190, 210), 1))

        # Baseline
        painter.drawLine(ml, y, w - mr, y)

        return y + 22

    def _draw_gene_track(self, painter, w, y, ml, mr):
        """Draw gene models (exon-intron structure)."""
        painter.setFont(QFont("Consolas", 7, QFont.Bold))
        track_h = 0
        rows = []  # list of (gene, row_idx)

        # Simple row packing to avoid overlaps
        row_ends = []  # rightmost x coordinate used in each row
        visible_genes = [g for g in self.genes
                         if g['chrom'] == self.chrom
                         and g['end'] >= self.view_start
                         and g['start'] <= self.view_end]

        for gene in visible_genes:
            gx1 = self._bp_to_x(gene['start'], w, ml, mr)
            gx2 = self._bp_to_x(gene['end'], w, ml, mr)
            # Find a row
            placed = False
            for ri, re in enumerate(row_ends):
                if gx1 > re + 5:
                    row_ends[ri] = gx2 + 40
                    rows.append((gene, ri))
                    placed = True
                    break
            if not placed:
                rows.append((gene, len(row_ends)))
                row_ends.append(gx2 + 40)

        n_rows = max(1, len(row_ends))
        row_h = 22

        # Track label
        painter.setPen(QColor(100, 110, 130))
        painter.setFont(QFont("Consolas", 7))
        painter.drawText(4, y, ml - 8, row_h, Qt.AlignRight | Qt.AlignVCenter, "Genes")

        for gene, ri in rows:
            gy = y + ri * row_h
            gx1 = self._bp_to_x(gene['start'], w, ml, mr)
            gx2 = self._bp_to_x(gene['end'], w, ml, mr)

            # Intron line
            mid_y = gy + row_h // 2
            painter.setPen(QPen(QColor(100, 140, 200), 1.2))
            painter.drawLine(int(max(ml, gx1)), mid_y, int(min(w - mr, gx2)), mid_y)

            # Strand arrow
            arrow_dir = 1 if gene.get('strand', '+') == '+' else -1
            ax = gx2 + 3 if arrow_dir > 0 else gx1 - 3
            if ml < ax < w - mr:
                painter.drawLine(int(ax), mid_y, int(ax - 4 * arrow_dir), mid_y - 3)
                painter.drawLine(int(ax), mid_y, int(ax - 4 * arrow_dir), mid_y + 3)

            # Exons
            for es, ee in gene.get('exons', []):
                ex1 = self._bp_to_x(es, w, ml, mr)
                ex2 = self._bp_to_x(ee, w, ml, mr)
                if ex2 < ml or ex1 > w - mr:
                    continue
                ex1 = max(ml, ex1)
                ex2 = min(w - mr, ex2)
                exon_w = max(2, ex2 - ex1)
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor(70, 130, 200, 200))
                painter.drawRoundedRect(int(ex1), mid_y - 5, int(exon_w), 10, 2, 2)

            # Gene name
            painter.setFont(QFont("Consolas", 7, QFont.Bold))
            painter.setPen(QColor(40, 70, 120))
            name_x = max(ml, gx1)
            painter.drawText(int(name_x), gy, 120, 12, Qt.AlignLeft, gene['name'])

        return y + n_rows * row_h

    def _draw_variant_track(self, painter, w, y, ml, mr):
        """Draw variant lollipop/tick marks."""
        visible = [v for v in self.variants
                   if v['chrom'] == self.chrom
                   and self.view_start <= v['pos'] <= self.view_end]

        track_h = 50
        # Track label
        painter.setFont(QFont("Consolas", 7))
        painter.setPen(QColor(100, 110, 130))
        painter.drawText(4, y, ml - 8, 14, Qt.AlignRight | Qt.AlignVCenter, "Variants")

        if not visible:
            painter.setPen(QColor(180, 190, 200))
            painter.drawText(ml, y, 200, track_h, Qt.AlignLeft | Qt.AlignVCenter, "(no variants in view)")
            return y + 20

        # Baseline
        baseline_y = y + track_h - 4
        painter.setPen(QPen(QColor(210, 215, 225), 1))
        painter.drawLine(ml, baseline_y, w - mr, baseline_y)

        for v in visible:
            vx = self._bp_to_x(v['pos'], w, ml, mr)
            if vx < ml or vx > w - mr:
                continue

            # Color by mode
            if self.color_by == 'impact':
                col = _qhex(IMPACT_COLORS.get(v.get('impact', 'MODIFIER'), 0x9E9E9E))
            elif self.color_by == 'type':
                col = _qhex(VARTYPE_COLORS.get(v.get('type', 'OTHER'), 0x9E9E9E))
            else:
                # Color by quality
                q = min(v.get('qual', 0) / 100.0, 1.0)
                col = QColor(int(255 * (1 - q)), int(200 * q), 50)

            # Lollipop stem
            stem_h = min(track_h - 8, max(12, v.get('qual', 30) / 100 * (track_h - 8)))
            painter.setPen(QPen(col, 1.2))
            painter.drawLine(int(vx), baseline_y, int(vx), int(baseline_y - stem_h))

            # Lollipop head
            head_r = 3 if v.get('type') == 'SNV' else 4
            painter.setPen(Qt.NoPen)
            painter.setBrush(col)
            if v.get('type') == 'DEL':
                # Triangle for deletions
                path = QPainterPath()
                path.moveTo(vx, baseline_y - stem_h - head_r)
                path.lineTo(vx - head_r, baseline_y - stem_h + head_r)
                path.lineTo(vx + head_r, baseline_y - stem_h + head_r)
                path.closeSubpath()
                painter.drawPath(path)
            elif v.get('type') == 'INS':
                # Diamond for insertions
                path = QPainterPath()
                path.moveTo(vx, baseline_y - stem_h - head_r)
                path.lineTo(vx + head_r, baseline_y - stem_h)
                path.lineTo(vx, baseline_y - stem_h + head_r)
                path.lineTo(vx - head_r, baseline_y - stem_h)
                path.closeSubpath()
                painter.drawPath(path)
            else:
                # Circle for SNVs
                painter.drawEllipse(int(vx - head_r), int(baseline_y - stem_h - head_r),
                                    head_r * 2, head_r * 2)

        # Count label
        painter.setFont(QFont("Consolas", 7))
        painter.setPen(QColor(120, 130, 150))
        painter.drawText(ml, baseline_y + 2, 200, 14, Qt.AlignLeft,
                         f"{len(visible)} variants in view")

        return y + track_h + 14

    def _draw_coverage_track(self, painter, w, y, ml, mr):
        """Draw coverage/depth plot."""
        cov_data = self.coverage.get(self.chrom, [])
        visible = [(p, d) for p, d in cov_data
                   if self.view_start <= p <= self.view_end]

        track_h = 60

        painter.setFont(QFont("Consolas", 7))
        painter.setPen(QColor(100, 110, 130))
        painter.drawText(4, y, ml - 8, 14, Qt.AlignRight | Qt.AlignVCenter, "Coverage")

        if not visible:
            return y + 20

        max_depth = max(d for _, d in visible) if visible else 1
        baseline_y = y + track_h

        # Area fill
        path = QPainterPath()
        path.moveTo(self._bp_to_x(visible[0][0], w, ml, mr), baseline_y)
        for pos, depth in visible:
            px = self._bp_to_x(pos, w, ml, mr)
            py = baseline_y - (depth / max_depth) * (track_h - 5)
            path.lineTo(px, py)
        path.lineTo(self._bp_to_x(visible[-1][0], w, ml, mr), baseline_y)
        path.closeSubpath()

        grad = QLinearGradient(0, y, 0, baseline_y)
        grad.setColorAt(0, QColor(76, 175, 80, 160))
        grad.setColorAt(1, QColor(76, 175, 80, 30))
        painter.setPen(Qt.NoPen)
        painter.setBrush(grad)
        painter.drawPath(path)

        # Outline
        painter.setPen(QPen(QColor(56, 142, 60), 1.2))
        painter.setBrush(Qt.NoBrush)
        outline = QPainterPath()
        outline.moveTo(self._bp_to_x(visible[0][0], w, ml, mr),
                       baseline_y - (visible[0][1] / max_depth) * (track_h - 5))
        for pos, depth in visible[1:]:
            outline.lineTo(self._bp_to_x(pos, w, ml, mr),
                           baseline_y - (depth / max_depth) * (track_h - 5))
        painter.drawPath(outline)

        # Y-axis label
        painter.setFont(QFont("Consolas", 7))
        painter.setPen(QColor(130, 140, 160))
        painter.drawText(ml - 55, y, 50, 12, Qt.AlignRight, f"{max_depth}x")
        painter.drawText(ml - 55, baseline_y - 10, 50, 12, Qt.AlignRight, "0x")

        return y + track_h + 4

    def _draw_sequence_track(self, painter, w, y, ml, mr):
        """Draw nucleotide sequence at high zoom."""
        seq_data = self.sequences.get(self.chrom)
        if not seq_data:
            return y

        seq_start = seq_data['start'] if isinstance(seq_data, dict) else 0
        seq = seq_data['seq'] if isinstance(seq_data, dict) else seq_data

        span = self.view_end - self.view_start
        draw_w = w - ml - mr
        bp_per_px = span / draw_w

        painter.setFont(QFont("Consolas", 7))
        painter.setPen(QColor(100, 110, 130))
        painter.drawText(4, y, ml - 8, 14, Qt.AlignRight | Qt.AlignVCenter, "Seq")

        nuc_w = max(8, draw_w / max(1, span))
        for bp in range(max(int(self.view_start), seq_start),
                        min(int(self.view_end), seq_start + len(seq))):
            idx = bp - seq_start
            if 0 <= idx < len(seq):
                nuc = seq[idx]
                nx = self._bp_to_x(bp, w, ml, mr)
                col = NUC_COLORS.get(nuc, 0x9E9E9E)
                painter.setPen(Qt.NoPen)
                painter.setBrush(_qhex(col))
                painter.drawRect(int(nx), y, max(1, int(nuc_w)), 16)
                if nuc_w > 6:
                    painter.setPen(QColor(255, 255, 255))
                    painter.setFont(QFont("Consolas", min(8, int(nuc_w * 0.8)), QFont.Bold))
                    painter.drawText(int(nx), y, max(1, int(nuc_w)), 16,
                                     Qt.AlignCenter, nuc.upper())

        return y + 22

    def _draw_legend(self, painter, w, h):
        """Draw color legend in bottom-right corner."""
        if not self.variants:
            return
        painter.setFont(QFont("Consolas", 7))
        if self.color_by == 'type':
            items = list(VARTYPE_COLORS.items())
            title = "Variant Type"
        elif self.color_by == 'impact':
            items = list(IMPACT_COLORS.items())
            title = "Impact"
        else:
            items = [("High Q", 0x00C832), ("Low Q", 0xFF3200)]
            title = "Quality"

        lx = w - 110
        ly = h - 14 - len(items) * 14

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 255, 255, 210))
        painter.drawRoundedRect(lx - 6, ly - 16, 110, len(items) * 14 + 22, 6, 6)

        painter.setFont(QFont("Consolas", 7, QFont.Bold))
        painter.setPen(QColor(60, 70, 90))
        painter.drawText(lx, ly - 14, 100, 12, Qt.AlignLeft, title)

        painter.setFont(QFont("Consolas", 7))
        for i, (label, col) in enumerate(items):
            iy = ly + i * 14
            painter.setPen(Qt.NoPen)
            painter.setBrush(_qhex(col))
            painter.drawEllipse(lx, iy + 2, 8, 8)
            painter.setPen(QColor(80, 90, 110))
            painter.drawText(lx + 14, iy, 90, 12, Qt.AlignLeft | Qt.AlignVCenter, label)

    # ── Mouse interaction ──

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._dragging = True
            self._lmx = e.x()
        e.accept()

    def mouseReleaseEvent(self, e):
        self._dragging = False
        e.accept()

    def mouseMoveEvent(self, e):
        if self._dragging:
            dx = e.x() - self._lmx
            span = self.view_end - self.view_start
            draw_w = self.width() - 80
            bp_shift = -dx / draw_w * span
            ch_len = CHROMOSOMES.get(self.chrom, {}).get('length', 1e9)
            new_start = max(0, min(ch_len - span, self.view_start + bp_shift))
            self.view_end = new_start + span
            self.view_start = new_start
            self._lmx = e.x()
        e.accept()

    def wheelEvent(self, e):
        factor = 0.8 if e.angleDelta().y() > 0 else 1.25
        center = (self.view_start + self.view_end) / 2
        half_span = (self.view_end - self.view_start) / 2 * factor
        half_span = max(50, min(CHROMOSOMES.get(self.chrom, {}).get('length', 3e9) / 2, half_span))
        ch_len = CHROMOSOMES.get(self.chrom, {}).get('length', 3e9)
        self.view_start = max(0, int(center - half_span))
        self.view_end = min(ch_len, int(center + half_span))
        e.accept()

# ═══════════════════════════════════════════════════════════════
#  GENOMELAB — MAIN API SINGLETON
# ═══════════════════════════════════════════════════════════════

class _Signals(QObject):
    status = Signal(str)
    load_done = Signal(str)

class GenomeLab:
    """
    Main genomics API. Registered as `genome` in the namespace.

    QUICK REFERENCE (for LLM context):
        genome.load_vcf(path)                  — load variants from VCF
        genome.load_fasta(path)                — load reference sequence
        genome.load_bed(path)                  — load BED regions
        genome.load_gff(path)                  — load gene annotations (GFF3/GTF)
        genome.load_demo(name)                 — load demo dataset: "brca"
        genome.goto(region_str)                — jump to "chr:start-end" or "chr", start, end
        genome.goto(chrom, start, end)         — jump to coordinates
        genome.set_chromosome(chrom)           — show full chromosome
        genome.zoom_in() / genome.zoom_out()   — 2x zoom
        genome.ideogram()                      — show full karyotype overlay
        genome.filter_variants(chrom=, min_qual=, gene=, impact=, vtype=)
        genome.variant_summary()               — print variant counts by type/chrom
        genome.annotate()                      — basic functional annotation
        genome.show_genes()                    — toggle gene track on
        genome.gene_info(name)                 — lookup gene details
        genome.show_sequence(chrom, start, end)— show nucleotide sequence
        genome.gc_content(chrom, start, end)   — compute GC%
        genome.find_motif(pattern)             — find sequence motif
        genome.translate(chrom, start, end)    — 6-frame translation
        genome.overlay_manhattan()             — Manhattan plot overlay
        genome.overlay_frequency()             — AF histogram overlay
        genome.overlay_qual()                  — quality distribution overlay
        genome.overlay_coverage()              — toggle coverage track
        genome.color_by(mode)                  — 'type', 'impact', or 'qual'
        genome.export_bed(path)                — export filtered regions
        genome.export_fasta(path, chrom, s, e) — export sequence
        genome.export_vcf(path)                — export filtered variants
        genome.stats()                         — genome-wide statistics
        genome.overlay(name, fn)               — custom QPainter overlay
        genome.remove_overlay(name)            — remove overlay
        genome.info                            — dict of properties
        genome.variants                        — current variant list
        genome.genes                           — current gene list
        genome.viewer                          — the GenomeViewer widget
        genome.log(msg)                        — append to log
    """

    def __init__(self, viewer, log_widget=None):
        self.viewer = viewer
        self._log = log_widget
        self.variants = []        # all loaded variants
        self._filtered = None     # filtered subset (None = all)
        self.genes = []
        self.sequences = {}       # {chrom: str or {start, seq}}
        self.bed_regions = []
        self.coverage = {}
        self.info = {}
        self._signals = _Signals()

    def log(self, msg):
        if self._log:
            self._log.append(f"[genome] {msg}")
        print(f"[genome] {msg}")

    # ── Loading ────────────────────────────────────────────────

    def load_demo(self, name="brca"):
        """Load a demo dataset for exploration."""
        key = name.lower().strip()
        if key not in DEMOS:
            self.log(f"Unknown demo: {key}. Available: {', '.join(DEMOS.keys())}")
            return self
        data = DEMOS[key]()
        self.variants = data.get('variants', [])
        self.genes = data.get('genes', [])
        self.coverage = data.get('coverage', {})
        self.sequences = data.get('sequences', {})
        self.info = {
            'name': f'Demo: {name}',
            'assembly': data.get('assembly', ''),
            'n_variants': len(self.variants),
            'n_genes': len(self.genes),
            'chromosomes': list(set(v['chrom'] for v in self.variants)),
            'source': 'demo',
        }
        self.viewer.variants = self.variants
        self.viewer.genes = self.genes
        self.viewer.coverage = self.coverage
        self.viewer.sequences = self.sequences
        # Navigate to first gene
        if self.genes:
            g = self.genes[0]
            pad = max(5000, (g['end'] - g['start']) * 0.2)
            self.viewer.set_region(g['chrom'], g['start'] - pad, g['end'] + pad)
        self._filtered = None
        self.log(f"Loaded demo '{name}': {len(self.variants)} variants, {len(self.genes)} genes")
        return self

    def load_vcf(self, path):
        """Load variants from VCF file."""
        path = os.path.expanduser(path)
        if not os.path.isfile(path):
            self.log(f"File not found: {path}"); return self
        self.log(f"Loading VCF: {path} ...")
        variants, headers, samples = parse_vcf(path)
        self.variants = variants
        self._filtered = None
        self.viewer.variants = variants
        # Variant type counts
        type_counts = defaultdict(int)
        for v in variants:
            type_counts[v.get('type', 'OTHER')] += 1
        chroms = sorted(set(v['chrom'] for v in variants))
        self.info.update({
            'filename': os.path.basename(path),
            'n_variants': len(variants),
            'variant_types': dict(type_counts),
            'chromosomes': chroms,
            'samples': samples,
            'source': 'vcf',
        })
        # Auto-navigate to first variant
        if variants:
            ch = variants[0]['chrom']
            ch_vars = [v for v in variants if v['chrom'] == ch]
            mn = min(v['pos'] for v in ch_vars)
            mx = max(v['pos'] for v in ch_vars)
            pad = max(5000, (mx - mn) * 0.1)
            self.viewer.set_region(ch, mn - pad, mx + pad)
        self.log(f"Loaded {len(variants)} variants from {os.path.basename(path)}")
        return self

    def load_fasta(self, path):
        """Load reference sequences from FASTA."""
        path = os.path.expanduser(path)
        if not os.path.isfile(path):
            self.log(f"File not found: {path}"); return self
        self.log(f"Loading FASTA: {path} ...")
        seqs = parse_fasta(path)
        self.sequences.update(seqs)
        self.viewer.sequences = self.sequences
        total_len = sum(len(s) for s in seqs.values())
        self.info.update({
            'n_sequences': len(seqs),
            'total_length': total_len,
            'seq_names': list(seqs.keys()),
        })
        self.log(f"Loaded {len(seqs)} sequences ({self.viewer._format_bp(total_len)})")
        return self

    def load_bed(self, path):
        """Load BED regions."""
        path = os.path.expanduser(path)
        if not os.path.isfile(path):
            self.log(f"File not found: {path}"); return self
        regions = parse_bed(path)
        self.bed_regions = regions
        self.viewer.bed_regions = regions
        self.log(f"Loaded {len(regions)} regions from BED")
        return self

    def load_gff(self, path):
        """Load gene annotations from GFF3/GTF."""
        path = os.path.expanduser(path)
        if not os.path.isfile(path):
            self.log(f"File not found: {path}"); return self
        self.log(f"Loading GFF: {path} ...")
        genes, features = parse_gff(path)
        self.genes = genes
        self.viewer.genes = genes
        self.info['n_genes'] = len(genes)
        self.info['genes'] = [g['name'] for g in genes]
        self.log(f"Loaded {len(genes)} genes, {len(features)} features")
        return self

    # ── Navigation ─────────────────────────────────────────────

    def goto(self, region_or_chrom, start=None, end=None):
        """Navigate to a genomic region.
        Usage:
            genome.goto("chr17:41196312-41277500")
            genome.goto("chr17", 41196312, 41277500)
            genome.goto("BRCA1")  # jump to gene by name
        """
        if start is not None and end is not None:
            self.viewer.set_region(str(region_or_chrom), int(start), int(end))
            self.log(f"Goto {region_or_chrom}:{start:,}-{end:,}")
            return self

        s = str(region_or_chrom).strip()
        # Try chr:start-end format
        m = re.match(r'(chr\w+):(\d[\d,]*)-(\d[\d,]*)', s.replace(',', ''))
        if m:
            self.viewer.set_region(m.group(1), int(m.group(2)), int(m.group(3)))
            self.log(f"Goto {m.group(1)}:{m.group(2)}-{m.group(3)}")
            return self

        # Try chromosome name
        if s in CHROMOSOMES:
            self.viewer.set_region(s, 0, CHROMOSOMES[s]['length'])
            self.log(f"Goto {s} (full chromosome)")
            return self

        # Try gene name
        for g in self.genes:
            if g['name'].upper() == s.upper():
                pad = max(5000, (g['end'] - g['start']) * 0.3)
                self.viewer.set_region(g['chrom'], g['start'] - pad, g['end'] + pad)
                self.log(f"Goto gene {g['name']}: {g['chrom']}:{g['start']:,}-{g['end']:,}")
                return self

        self.log(f"Could not parse region: {s}")
        return self

    def set_chromosome(self, chrom):
        """Show a full chromosome."""
        ch = chrom if chrom.startswith('chr') else f'chr{chrom}'
        if ch in CHROMOSOMES:
            self.viewer.set_region(ch, 0, CHROMOSOMES[ch]['length'])
            self.log(f"Showing full {ch}")
        else:
            self.log(f"Unknown chromosome: {chrom}")
        return self

    def zoom_in(self, factor=2.0):
        """Zoom into center of current view."""
        center = (self.viewer.view_start + self.viewer.view_end) / 2
        half = (self.viewer.view_end - self.viewer.view_start) / (2 * factor)
        self.viewer.view_start = max(0, int(center - half))
        self.viewer.view_end = int(center + half)
        return self

    def zoom_out(self, factor=2.0):
        """Zoom out from center of current view."""
        center = (self.viewer.view_start + self.viewer.view_end) / 2
        half = (self.viewer.view_end - self.viewer.view_start) * factor / 2
        ch_len = CHROMOSOMES.get(self.viewer.chrom, {}).get('length', 3e9)
        self.viewer.view_start = max(0, int(center - half))
        self.viewer.view_end = min(int(ch_len), int(center + half))
        return self

    # ── Variant analysis ───────────────────────────────────────

    def filter_variants(self, chrom=None, min_qual=None, max_qual=None,
                        gene=None, impact=None, vtype=None, filter_pass=None,
                        min_af=None, max_af=None):
        """Filter variants. Returns self for chaining. Filtered set stored internally."""
        result = list(self.variants)
        if chrom:
            ch = chrom if chrom.startswith('chr') else f'chr{chrom}'
            result = [v for v in result if v['chrom'] == ch]
        if min_qual is not None:
            result = [v for v in result if v.get('qual', 0) >= min_qual]
        if max_qual is not None:
            result = [v for v in result if v.get('qual', 0) <= max_qual]
        if gene:
            result = [v for v in result if v.get('gene', '').upper() == gene.upper()]
        if impact:
            result = [v for v in result if v.get('impact', '').upper() == impact.upper()]
        if vtype:
            result = [v for v in result if v.get('type', '').upper() == vtype.upper()]
        if filter_pass:
            result = [v for v in result if v.get('filter') == 'PASS']
        if min_af is not None:
            result = [v for v in result if v.get('af', 0) >= min_af]
        if max_af is not None:
            result = [v for v in result if v.get('af', 1) <= max_af]
        self._filtered = result
        self.viewer.variants = result
        self.log(f"Filtered: {len(result)}/{len(self.variants)} variants")
        return self

    def reset_filter(self):
        """Remove all filters, show all variants."""
        self._filtered = None
        self.viewer.variants = self.variants
        self.log("Filters cleared")
        return self

    def variant_summary(self):
        """Print variant summary statistics."""
        vlist = self._filtered if self._filtered is not None else self.variants
        if not vlist:
            self.log("No variants loaded"); return self

        self.log(f"─── Variant Summary ({len(vlist)} variants) ───")
        # By type
        tc = defaultdict(int)
        for v in vlist: tc[v.get('type', 'OTHER')] += 1
        self.log("By type: " + ", ".join(f"{k}={v}" for k, v in sorted(tc.items())))
        # By impact
        ic = defaultdict(int)
        for v in vlist: ic[v.get('impact', 'MODIFIER')] += 1
        self.log("By impact: " + ", ".join(f"{k}={v}" for k, v in sorted(ic.items())))
        # By chromosome
        cc = defaultdict(int)
        for v in vlist: cc[v['chrom']] += 1
        top_chroms = sorted(cc.items(), key=lambda x: -x[1])[:8]
        self.log("By chrom: " + ", ".join(f"{k}={v}" for k, v in top_chroms))
        # Quality stats
        quals = [v.get('qual', 0) for v in vlist]
        self.log(f"Quality: min={min(quals):.1f}, median={np.median(quals):.1f}, max={max(quals):.1f}")
        # PASS rate
        n_pass = sum(1 for v in vlist if v.get('filter') == 'PASS')
        self.log(f"PASS rate: {n_pass}/{len(vlist)} ({100*n_pass/len(vlist):.1f}%)")
        return self

    def annotate(self):
        """Basic functional annotation: mark variants overlapping gene exons."""
        if not self.genes:
            self.log("No gene annotations loaded"); return self

        # Build exon index
        exon_index = defaultdict(list)  # chrom → [(start, end, gene_name)]
        for gene in self.genes:
            for es, ee in gene.get('exons', []):
                exon_index[gene['chrom']].append((es, ee, gene['name']))

        annotated = 0
        for v in self.variants:
            ch = v['chrom']
            if ch in exon_index:
                for es, ee, gname in exon_index[ch]:
                    if es <= v['pos'] <= ee:
                        v['gene'] = gname
                        # Assign higher impact if in exon
                        if v.get('impact') == 'MODIFIER':
                            v['impact'] = 'MODERATE'
                        annotated += 1
                        break
        self.log(f"Annotated {annotated} variants with gene/exon overlap")
        return self

    # ── Sequence operations ────────────────────────────────────

    def get_sequence(self, chrom, start, end):
        """Get sequence string for a region. Returns None if not loaded."""
        seq_data = self.sequences.get(chrom)
        if seq_data is None:
            return None
        if isinstance(seq_data, dict):
            seq_start = seq_data['start']
            seq = seq_data['seq']
        else:
            seq_start = 0
            seq = seq_data
        s = start - seq_start
        e = end - seq_start
        if s < 0 or e > len(seq):
            return None
        return seq[s:e]

    def show_sequence(self, chrom=None, start=None, end=None):
        """Show nucleotide sequence in the viewer. Zooms in if needed."""
        if chrom and start and end:
            self.viewer.set_region(chrom, start, end)
        self.viewer.show_sequence = True
        span = self.viewer.view_end - self.viewer.view_start
        if span > 500:
            self.log("Zoom in to < 500 bp to see individual nucleotides")
            center = (self.viewer.view_start + self.viewer.view_end) // 2
            self.viewer.set_region(self.viewer.chrom, center - 200, center + 200)
        return self

    def gc_content(self, chrom=None, start=None, end=None):
        """Compute GC content (%) in a region."""
        ch = chrom or self.viewer.chrom
        s = start or self.viewer.view_start
        e = end or self.viewer.view_end
        seq = self.get_sequence(ch, s, e)
        if seq is None:
            self.log(f"No sequence data for {ch}:{s}-{e}"); return None
        seq_upper = seq.upper()
        gc = sum(1 for c in seq_upper if c in ('G', 'C'))
        total = sum(1 for c in seq_upper if c in ('A', 'T', 'G', 'C'))
        pct = 100.0 * gc / total if total > 0 else 0
        self.log(f"GC content {ch}:{s:,}-{e:,}: {pct:.1f}% ({gc}/{total} bp)")
        return pct

    def find_motif(self, pattern, chrom=None):
        """Find occurrences of a sequence motif (regex-capable)."""
        ch = chrom or self.viewer.chrom
        seq_data = self.sequences.get(ch)
        if seq_data is None:
            self.log(f"No sequence for {ch}"); return []
        seq = seq_data['seq'] if isinstance(seq_data, dict) else seq_data
        offset = seq_data['start'] if isinstance(seq_data, dict) else 0
        matches = []
        for m in re.finditer(pattern.upper(), seq.upper()):
            matches.append({"start": offset + m.start(), "end": offset + m.end(),
                            "seq": m.group()})
        self.log(f"Found {len(matches)} matches for '{pattern}' on {ch}")
        if matches:
            # Navigate to first match
            self.viewer.set_region(ch, matches[0]['start'] - 50, matches[0]['end'] + 50)
        return matches

    def translate(self, chrom=None, start=None, end=None, frame=0):
        """Translate a region in specified reading frame (0, 1, 2). Returns amino acid string."""
        ch = chrom or self.viewer.chrom
        s = start or self.viewer.view_start
        e = end or self.viewer.view_end
        seq = self.get_sequence(ch, s, e)
        if seq is None:
            self.log(f"No sequence for {ch}:{s}-{e}"); return ""
        seq = seq.upper()[frame:]
        codons = [seq[i:i+3] for i in range(0, len(seq) - 2, 3)]
        aa = ''.join(CODON_TABLE.get(c, '?') for c in codons)
        self.log(f"Translation frame +{frame}: {aa[:60]}{'...' if len(aa) > 60 else ''}")
        return aa

    def reverse_complement(self, seq):
        """Return reverse complement of a DNA sequence."""
        return ''.join(COMPLEMENT.get(c, 'N') for c in reversed(seq))

    # ── Gene operations ────────────────────────────────────────

    def show_genes(self, show=True):
        """Toggle gene track visibility."""
        self.viewer.show_genes = show
        return self

    def gene_info(self, name):
        """Lookup gene details by name."""
        for g in self.genes:
            if g['name'].upper() == name.upper():
                self.log(f"─── {g['name']} ───")
                self.log(f"  Location: {g['chrom']}:{g['start']:,}-{g['end']:,} ({g.get('strand', '?')})")
                self.log(f"  Size: {g['end'] - g['start']:,} bp")
                self.log(f"  Exons: {len(g.get('exons', []))}")
                self.log(f"  Biotype: {g.get('biotype', '?')}")
                if g.get('description'):
                    self.log(f"  Description: {g['description']}")
                # Count variants in this gene
                gene_vars = [v for v in self.variants
                             if v.get('gene', '').upper() == name.upper()]
                if gene_vars:
                    self.log(f"  Variants: {len(gene_vars)}")
                return g
        self.log(f"Gene not found: {name}")
        return None

    # ── Overlays ───────────────────────────────────────────────

    def overlay_manhattan(self, width=None, height=180):
        """Show Manhattan plot as overlay (variants by chromosome, y = -log10(p) ≈ qual/10)."""
        vlist = self._filtered if self._filtered is not None else self.variants
        if not vlist:
            self.log("No variants for Manhattan plot"); return self

        def draw_manhattan(painter, w, h):
            ow = width or w - 80
            oh = height
            ox, oy = 50, h - oh - 20

            # Background
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(255, 255, 255, 220))
            painter.drawRoundedRect(ox - 10, oy - 20, ow + 20, oh + 30, 8, 8)

            painter.setFont(QFont("Consolas", 8, QFont.Bold))
            painter.setPen(QColor(50, 60, 80))
            painter.drawText(ox, oy - 18, ow, 14, Qt.AlignCenter, "Manhattan Plot")

            # Compute positions
            chrom_offsets = {}
            cum = 0
            for ch_name, ch_data in CHROMOSOMES.items():
                chrom_offsets[ch_name] = cum
                cum += ch_data['length']

            # Axes
            painter.setPen(QPen(QColor(200, 205, 215), 1))
            painter.drawLine(ox, oy + oh, ox + ow, oy + oh)
            painter.drawLine(ox, oy, ox, oy + oh)

            # Y axis: -log10(qual/max_qual) as proxy for significance
            max_score = max(v.get('qual', 1) for v in vlist)
            if max_score <= 0: max_score = 1

            # Plot points
            for v in vlist:
                ch_off = chrom_offsets.get(v['chrom'], 0)
                genome_pos = ch_off + v['pos']
                px = ox + genome_pos / cum * ow
                score = v.get('qual', 0) / max_score
                py = oy + oh - score * (oh - 10)

                ch_idx = list(CHROMOSOMES.keys()).index(v['chrom']) if v['chrom'] in CHROMOSOMES else 0
                col = _qhex(CHROMOSOMES[v['chrom']]['color']) if v['chrom'] in CHROMOSOMES else QColor(150, 150, 150)
                col.setAlpha(180)
                painter.setPen(Qt.NoPen)
                painter.setBrush(col)
                painter.drawEllipse(int(px) - 2, int(py) - 2, 4, 4)

            # Chromosome labels
            painter.setFont(QFont("Consolas", 6))
            painter.setPen(QColor(130, 140, 160))
            for i, (ch_name, ch_data) in enumerate(CHROMOSOMES.items()):
                if i % 2 == 0:
                    cx = ox + (chrom_offsets[ch_name] + ch_data['length'] / 2) / cum * ow
                    label = ch_name.replace('chr', '')
                    painter.drawText(int(cx) - 8, oy + oh + 2, 16, 10, Qt.AlignCenter, label)

            # Significance line
            sig_y = oy + oh - 0.5 * (oh - 10)  # at qual = 50% of max
            painter.setPen(QPen(QColor(220, 80, 80, 120), 1, Qt.DashLine))
            painter.drawLine(ox, int(sig_y), ox + ow, int(sig_y))

        self.viewer.add_overlay('manhattan', draw_manhattan)
        self.log("Manhattan plot overlay added")
        return self

    def overlay_frequency(self, width=250, height=150):
        """Show allele frequency histogram as overlay."""
        vlist = self._filtered if self._filtered is not None else self.variants
        afs = [v.get('af', 0) for v in vlist if v.get('af', 0) > 0]
        if not afs:
            self.log("No AF data available"); return self

        bins = np.histogram(afs, bins=20, range=(0, 1))

        def draw_af(painter, w, h):
            ox, oy = w - width - 20, 30
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(255, 255, 255, 220))
            painter.drawRoundedRect(ox - 6, oy - 16, width + 12, height + 24, 8, 8)

            painter.setFont(QFont("Consolas", 8, QFont.Bold))
            painter.setPen(QColor(50, 60, 80))
            painter.drawText(ox, oy - 14, width, 12, Qt.AlignCenter, "Allele Frequency")

            counts, edges = bins
            max_c = max(counts) if max(counts) > 0 else 1
            bar_w = (width - 20) / len(counts)

            for i, c in enumerate(counts):
                bh = c / max_c * (height - 30)
                bx = ox + 10 + i * bar_w
                by = oy + height - 15 - bh
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor(92, 107, 192, 200))
                painter.drawRect(int(bx), int(by), max(1, int(bar_w - 1)), int(bh))

            painter.setFont(QFont("Consolas", 6))
            painter.setPen(QColor(130, 140, 160))
            painter.drawText(ox + 10, oy + height - 12, 20, 10, Qt.AlignLeft, "0")
            painter.drawText(ox + width - 30, oy + height - 12, 20, 10, Qt.AlignRight, "1.0")
            painter.drawText(ox + 10, oy + height - 4, width - 20, 10, Qt.AlignCenter, "AF")

        self.viewer.add_overlay('af_hist', draw_af)
        self.log("AF histogram overlay added")
        return self

    def overlay_qual(self, width=250, height=150):
        """Show quality score distribution as overlay."""
        vlist = self._filtered if self._filtered is not None else self.variants
        quals = [v.get('qual', 0) for v in vlist]
        if not quals:
            self.log("No quality data"); return self

        bins = np.histogram(quals, bins=25)

        def draw_qual(painter, w, h):
            ox, oy = w - width - 20, 30
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(255, 255, 255, 220))
            painter.drawRoundedRect(ox - 6, oy - 16, width + 12, height + 24, 8, 8)

            painter.setFont(QFont("Consolas", 8, QFont.Bold))
            painter.setPen(QColor(50, 60, 80))
            painter.drawText(ox, oy - 14, width, 12, Qt.AlignCenter, "Quality Distribution")

            counts, edges = bins
            max_c = max(counts) if max(counts) > 0 else 1
            bar_w = (width - 20) / len(counts)

            for i, c in enumerate(counts):
                bh = c / max_c * (height - 30)
                bx = ox + 10 + i * bar_w
                by = oy + height - 15 - bh
                q_frac = edges[i] / (max(edges) if max(edges) > 0 else 1)
                col = QColor(int(255 * (1 - q_frac)), int(200 * q_frac), 50, 200)
                painter.setPen(Qt.NoPen)
                painter.setBrush(col)
                painter.drawRect(int(bx), int(by), max(1, int(bar_w - 1)), int(bh))

            painter.setFont(QFont("Consolas", 6))
            painter.setPen(QColor(130, 140, 160))
            painter.drawText(ox + 10, oy + height - 4, width - 20, 10, Qt.AlignCenter, "QUAL")

        self.viewer.add_overlay('qual_hist', draw_qual)
        self.log("Quality distribution overlay added")
        return self

    def overlay_coverage(self, show=True):
        """Toggle coverage track."""
        self.viewer.show_coverage = show
        self.log(f"Coverage track {'on' if show else 'off'}")
        return self

    def color_by_mode(self, mode):
        """Set variant coloring mode: 'type', 'impact', or 'qual'."""
        if mode in ('type', 'impact', 'qual'):
            self.viewer.color_by = mode
            self.log(f"Color by: {mode}")
        else:
            self.log(f"Unknown color mode: {mode}. Use 'type', 'impact', or 'qual'")
        return self

    def ideogram(self):
        """Show full karyotype ideogram as overlay."""
        def draw_karyotype(painter, w, h):
            ox, oy = 60, 30
            ow = w - 80
            n_chroms = len(CHROMOSOMES)
            max_len = max(c['length'] for c in CHROMOSOMES.values())
            bar_h = max(8, min(18, (h - 80) / n_chroms - 2))
            spacing = bar_h + 3

            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(255, 255, 255, 230))
            painter.drawRoundedRect(ox - 10, oy - 20, ow + 20, n_chroms * spacing + 30, 8, 8)

            painter.setFont(QFont("Consolas", 8, QFont.Bold))
            painter.setPen(QColor(50, 60, 80))
            painter.drawText(ox, oy - 18, ow, 14, Qt.AlignCenter, "Human Karyotype (GRCh38)")

            for i, (ch_name, ch_data) in enumerate(CHROMOSOMES.items()):
                cy = oy + i * spacing
                ch_w = ch_data['length'] / max_len * (ow - 50)

                # Label
                painter.setFont(QFont("Consolas", 7))
                painter.setPen(QColor(100, 110, 130))
                painter.drawText(ox - 8, cy, 40, int(bar_h), Qt.AlignRight | Qt.AlignVCenter,
                                 ch_name.replace('chr', ''))

                # Bar
                bx = ox + 40
                painter.setPen(QPen(QColor(200, 205, 215), 0.5))
                painter.setBrush(_qhex(ch_data['color']))
                painter.drawRoundedRect(bx, cy, int(ch_w), int(bar_h), 3, 3)

                # Centromere notch
                cs, ce = ch_data['centromere']
                cx1 = bx + cs / ch_data['length'] * ch_w
                cx2 = bx + ce / ch_data['length'] * ch_w
                painter.setBrush(QColor(60, 60, 80, 120))
                painter.setPen(Qt.NoPen)
                painter.drawRect(int(cx1), cy, max(1, int(cx2 - cx1)), int(bar_h))

                # Variant count on this chrom
                n_vars = sum(1 for v in self.variants if v['chrom'] == ch_name)
                if n_vars > 0:
                    painter.setFont(QFont("Consolas", 6))
                    painter.setPen(QColor(180, 60, 60))
                    painter.drawText(int(bx + ch_w + 4), cy, 50, int(bar_h),
                                     Qt.AlignLeft | Qt.AlignVCenter, f"{n_vars}")

        self.viewer.add_overlay('karyotype', draw_karyotype)
        self.log("Karyotype ideogram overlay added")
        return self

    # ── Overlay management ─────────────────────────────────────

    def overlay(self, name, fn):
        """Add custom overlay: fn(painter, width, height)."""
        self.viewer.add_overlay(name, fn)
        return self

    def remove_overlay(self, name):
        self.viewer.remove_overlay(name)
        return self

    def clear_overlays(self):
        """Remove all overlays."""
        self.viewer._overlays.clear()
        return self

    # ── Export ─────────────────────────────────────────────────

    def export_vcf(self, path="~/filtered.vcf"):
        """Export current (filtered) variants as VCF."""
        path = os.path.expanduser(path)
        vlist = self._filtered if self._filtered is not None else self.variants
        with open(path, 'w') as f:
            f.write("##fileformat=VCFv4.3\n")
            f.write("##source=GenomeLab\n")
            f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
            for v in vlist:
                info_parts = [f"TYPE={v.get('type','.')}"]
                if v.get('af'): info_parts.append(f"AF={v['af']}")
                if v.get('gene') and v['gene'] != '.': info_parts.append(f"GENE={v['gene']}")
                if v.get('impact'): info_parts.append(f"IMPACT={v['impact']}")
                info_str = ';'.join(info_parts)
                f.write(f"{v['chrom']}\t{v['pos']}\t{v.get('id','.')}\t{v['ref']}\t{v['alt']}\t"
                        f"{v['qual']}\t{v.get('filter','.')}\t{info_str}\n")
        self.log(f"Exported {len(vlist)} variants to {path}")
        return path

    def export_bed(self, path="~/regions.bed"):
        """Export current view or filtered regions as BED."""
        path = os.path.expanduser(path)
        vlist = self._filtered if self._filtered is not None else self.variants
        with open(path, 'w') as f:
            for v in vlist:
                end = v['pos'] + max(len(v.get('ref', '')), 1)
                name = f"{v.get('type','.')}_{v.get('gene','.')}"
                f.write(f"{v['chrom']}\t{v['pos']}\t{end}\t{name}\t{v.get('qual',0)}\n")
        self.log(f"Exported {len(vlist)} regions to {path}")
        return path

    def export_fasta(self, path="~/region.fa", chrom=None, start=None, end=None):
        """Export sequence region as FASTA."""
        path = os.path.expanduser(path)
        ch = chrom or self.viewer.chrom
        s = start or self.viewer.view_start
        e = end or self.viewer.view_end
        seq = self.get_sequence(ch, s, e)
        if seq is None:
            self.log(f"No sequence for {ch}:{s}-{e}"); return None
        with open(path, 'w') as f:
            f.write(f">{ch}:{s}-{e}\n")
            for i in range(0, len(seq), 80):
                f.write(seq[i:i+80] + '\n')
        self.log(f"Exported {len(seq)} bp to {path}")
        return path

    def stats(self):
        """Print genome-wide statistics."""
        self.log("═══ Genome Statistics ═══")
        self.log(f"Assembly: {self.info.get('assembly', 'unknown')}")
        self.log(f"Variants: {len(self.variants)}")
        if self._filtered is not None:
            self.log(f"Filtered: {len(self._filtered)}")
        self.log(f"Genes: {len(self.genes)}")
        self.log(f"Sequences loaded: {len(self.sequences)}")
        if self.sequences:
            total = sum(len(s['seq']) if isinstance(s, dict) else len(s)
                        for s in self.sequences.values())
            self.log(f"Total sequence: {self.viewer._format_bp(total)}")
        self.log(f"BED regions: {len(self.bed_regions)}")
        self.log(f"Coverage tracks: {len(self.coverage)} chromosomes")
        self.log(f"Current view: {self.viewer.chrom}:{self.viewer.view_start:,}-{self.viewer.view_end:,}")
        return self


# ═══════════════════════════════════════════════════════════════
#  LIGHT-MODE STYLESHEET
# ═══════════════════════════════════════════════════════════════

_SS = """
QWidget{background:rgba(255,255,255,220);color:#2a2e3a;font-family:'Consolas','Menlo',monospace;font-size:11px}
QPushButton{background:rgba(245,247,250,240);border:1px solid #d0d5e0;border-radius:5px;padding:6px 10px;color:#2e7d32;font-weight:bold;font-size:10px}
QPushButton:hover{background:rgba(230,248,238,250);border-color:#81c784}
QPushButton:checked{background:rgba(76,175,80,30);border-color:#66bb6a;color:#2e7d32}
QSlider::groove:horizontal{height:4px;background:#d8dce6;border-radius:2px}
QSlider::handle:horizontal{background:#66bb6a;width:14px;margin:-5px 0;border-radius:7px}
QComboBox{background:rgba(250,251,253,240);border:1px solid #d0d5e0;border-radius:4px;padding:5px 8px}
QComboBox QAbstractItemView{background:white;border:1px solid #d0d5e0;selection-background-color:#e8f5e9}
QTextEdit{background:rgba(250,251,253,220);border:1px solid #dce0e8;border-radius:4px;font-size:10px;color:#4a5568;padding:6px}
QCheckBox{spacing:8px;color:#4a5568}
QTabWidget::pane{border:1px solid #d8dce6;background:rgba(255,255,255,200);border-top:none}
QTabBar::tab{background:rgba(240,242,246,200);color:#6a7a8a;padding:7px 14px;font-size:10px;font-weight:bold;border:1px solid #d8dce6;border-bottom:none}
QTabBar::tab:selected{background:rgba(255,255,255,240);color:#2e7d32;border-bottom:2px solid #66bb6a}
QListWidget{background:rgba(250,251,253,220);border:1px solid #dce0e8;border-radius:4px;font-size:10px;color:#4a5568}
QListWidget::item:selected{background:#e8f5e9;color:#1b5e20}
QLabel{background:transparent}
QScrollArea{border:none;background:transparent}
QLineEdit{background:rgba(250,251,253,240);border:1px solid #d0d5e0;border-radius:4px;padding:5px 8px}
QProgressBar{border:1px solid #d0d5e0;border-radius:4px;text-align:center;background:rgba(245,247,250,240);font-size:9px}
QProgressBar::chunk{background:#66bb6a;border-radius:3px}
"""

def _lbl(text):
    l = QLabel(text.upper())
    l.setStyleSheet("font-size:9px;letter-spacing:1.5px;color:#8a96a8;font-weight:bold;padding:2px 0;background:transparent")
    return l


# ═══════════════════════════════════════════════════════════════
#  GENOME LAB APP — UI ASSEMBLY, SINGLETON & SIGNAL WIRING
# ═══════════════════════════════════════════════════════════════

class GenomeLabApp:
    """Encapsulates all UI assembly, singleton creation, and signal wiring."""

    def __init__(self):
        # ── Main widget ──
        self.main_widget = QWidget()
        self.main_widget.setAttribute(Qt.WA_TranslucentBackground, True)
        self.main_layout = QHBoxLayout(self.main_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # ── Panel ──
        self.panel = QWidget()
        self.panel.setFixedWidth(290)
        self.panel.setStyleSheet(_SS)
        self.panel.setAttribute(Qt.WA_TranslucentBackground, True)
        self.ps = QScrollArea()
        self.ps.setWidgetResizable(True)
        self.ps.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.ps.setStyleSheet("QScrollArea{border:none;background:transparent}QScrollBar:vertical{width:5px;background:transparent}QScrollBar::handle:vertical{background:#c0c8d4;border-radius:2px;min-height:30px}")
        self.inner = QWidget()
        self.inner.setAttribute(Qt.WA_TranslucentBackground, True)
        self.lay = QVBoxLayout(self.inner)
        self.lay.setSpacing(4)
        self.lay.setContentsMargins(10, 10, 10, 10)

        self._build_header()
        self._build_tabs()

        self.lay.addWidget(self.tabs)
        self.ps.setWidget(self.inner)
        playout = QVBoxLayout(self.panel)
        playout.setContentsMargins(0, 0, 0, 0)
        playout.addWidget(self.ps)

        self.viewer = GenomeViewer()
        self.viewer.setStyleSheet("background:transparent")
        self.main_layout.addWidget(self.panel)
        self.main_layout.addWidget(self.viewer, 1)

        # ── Create singleton & wire signals ──
        self.genome = GenomeLab(self.viewer, self.log_edit)
        self.gen = self.genome  # alias

        self._wrap_methods()
        self._wire_signals()

        # Load demo on startup
        self.genome.load_demo("brca")

    # ── Header ──

    def _build_header(self):
        hdr = QWidget()
        hdr.setAttribute(Qt.WA_TranslucentBackground, True)
        hl = QHBoxLayout(hdr)
        hl.setContentsMargins(0, 0, 0, 4)
        ic = QLabel("\U0001f9ec")  # 🧬
        ic.setStyleSheet("font-size:20px;background:rgba(232,245,233,200);border:1px solid #c8e6c9;border-radius:7px;padding:3px 7px")
        nw = QWidget()
        nw.setAttribute(Qt.WA_TranslucentBackground, True)
        nl = QVBoxLayout(nw)
        nl.setContentsMargins(6, 0, 0, 0)
        nl.setSpacing(0)
        _title_lbl = QLabel("GenomeLab")
        _title_lbl.setStyleSheet("font-size:14px;font-weight:bold;color:#1b5e20;background:transparent")
        _sub_lbl = QLabel("GENOMICS WORKBENCH")
        _sub_lbl.setStyleSheet("font-size:7px;letter-spacing:2px;color:#8a96a8;background:transparent")
        nl.addWidget(_title_lbl)
        nl.addWidget(_sub_lbl)
        hl.addWidget(ic)
        hl.addWidget(nw)
        hl.addStretch()
        self.lay.addWidget(hdr)

    # ── Tabs ──

    def _build_tabs(self):
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(_SS)

        self._build_tab_navigate()
        self._build_tab_variants()
        self._build_tab_analysis()
        self._build_tab_info()
        self._build_tab_import()

    def _build_tab_navigate(self):
        t1 = QWidget()
        t1.setAttribute(Qt.WA_TranslucentBackground, True)
        t1l = QVBoxLayout(t1)
        t1l.setSpacing(5)
        t1l.setContentsMargins(6, 8, 6, 6)

        t1l.addWidget(_lbl("Chromosome"))
        self.chrom_combo = QComboBox()
        self.chrom_combo.addItems(list(CHROMOSOMES.keys()))
        self.chrom_combo.setCurrentText("chr17")
        t1l.addWidget(self.chrom_combo)

        t1l.addWidget(_lbl("Region (chr:start-end)"))
        self.region_edit = QLineEdit()
        self.region_edit.setPlaceholderText("chr17:41196312-41277500")
        t1l.addWidget(self.region_edit)

        nav_w = QWidget()
        nav_w.setAttribute(Qt.WA_TranslucentBackground, True)
        nav_l = QHBoxLayout(nav_w)
        nav_l.setContentsMargins(0, 0, 0, 0)
        nav_l.setSpacing(2)
        self.zoom_in_btn = QPushButton("Zoom +")
        self.zoom_out_btn = QPushButton("Zoom −")
        nav_l.addWidget(self.zoom_in_btn)
        nav_l.addWidget(self.zoom_out_btn)
        t1l.addWidget(nav_w)

        t1l.addWidget(_lbl("Display"))
        self.show_genes_cb = QCheckBox("Gene Track")
        self.show_genes_cb.setChecked(True)
        t1l.addWidget(self.show_genes_cb)
        self.show_variants_cb = QCheckBox("Variant Track")
        self.show_variants_cb.setChecked(True)
        t1l.addWidget(self.show_variants_cb)
        self.show_cov_cb = QCheckBox("Coverage Track")
        self.show_cov_cb.setChecked(False)
        t1l.addWidget(self.show_cov_cb)
        self.show_seq_cb = QCheckBox("Sequence (zoom in)")
        self.show_seq_cb.setChecked(False)
        t1l.addWidget(self.show_seq_cb)

        t1l.addWidget(_lbl("Color Variants By"))
        self.color_combo = QComboBox()
        self.color_combo.addItems(["type", "impact", "qual"])
        t1l.addWidget(self.color_combo)

        t1l.addStretch()
        self.tabs.addTab(t1, "Navigate")

    def _build_tab_variants(self):
        t2 = QWidget()
        t2.setAttribute(Qt.WA_TranslucentBackground, True)
        t2l = QVBoxLayout(t2)
        t2l.setSpacing(5)
        t2l.setContentsMargins(6, 8, 6, 6)

        t2l.addWidget(_lbl("Filter"))
        self.filter_gene_edit = QLineEdit()
        self.filter_gene_edit.setPlaceholderText("Gene name (e.g. BRCA1)")
        t2l.addWidget(self.filter_gene_edit)

        self.filter_impact_combo = QComboBox()
        self.filter_impact_combo.addItems(["All Impacts", "HIGH", "MODERATE", "LOW", "MODIFIER"])
        t2l.addWidget(self.filter_impact_combo)

        self.filter_type_combo = QComboBox()
        self.filter_type_combo.addItems(["All Types", "SNV", "INS", "DEL", "MNV"])
        t2l.addWidget(self.filter_type_combo)

        t2l.addWidget(_lbl("Min Quality"))
        self.qual_sl = QSlider(Qt.Horizontal)
        self.qual_sl.setRange(0, 200)
        self.qual_sl.setValue(0)
        t2l.addWidget(self.qual_sl)

        filter_w = QWidget()
        filter_w.setAttribute(Qt.WA_TranslucentBackground, True)
        filter_l = QHBoxLayout(filter_w)
        filter_l.setContentsMargins(0, 0, 0, 0)
        filter_l.setSpacing(2)
        self.apply_filter_btn = QPushButton("Apply Filter")
        self.reset_filter_btn = QPushButton("Reset")
        filter_l.addWidget(self.apply_filter_btn)
        filter_l.addWidget(self.reset_filter_btn)
        t2l.addWidget(filter_w)

        t2l.addWidget(_lbl("Variant List"))
        self.var_list = QListWidget()
        self.var_list.setMinimumHeight(120)
        t2l.addWidget(self.var_list)

        t2l.addStretch()
        self.tabs.addTab(t2, "Variants")

    def _build_tab_analysis(self):
        t3 = QWidget()
        t3.setAttribute(Qt.WA_TranslucentBackground, True)
        t3l = QVBoxLayout(t3)
        t3l.setSpacing(5)
        t3l.setContentsMargins(6, 8, 6, 6)

        t3l.addWidget(_lbl("Overlays"))
        self.manhattan_btn = QPushButton("Manhattan Plot")
        t3l.addWidget(self.manhattan_btn)
        self.af_btn = QPushButton("AF Distribution")
        t3l.addWidget(self.af_btn)
        self.qual_btn = QPushButton("Quality Dist.")
        t3l.addWidget(self.qual_btn)
        self.karyo_btn = QPushButton("Karyotype")
        t3l.addWidget(self.karyo_btn)
        self.clear_ov_btn = QPushButton("Clear Overlays")
        t3l.addWidget(self.clear_ov_btn)

        t3l.addWidget(_lbl("Motif Search"))
        self.motif_edit = QLineEdit()
        self.motif_edit.setPlaceholderText("e.g. GAATTC (EcoRI)")
        t3l.addWidget(self.motif_edit)
        self.motif_btn = QPushButton("Find Motif")
        t3l.addWidget(self.motif_btn)

        t3l.addWidget(_lbl("Export"))
        exp_w = QWidget()
        exp_w.setAttribute(Qt.WA_TranslucentBackground, True)
        exp_l = QHBoxLayout(exp_w)
        exp_l.setContentsMargins(0, 0, 0, 0)
        exp_l.setSpacing(2)
        self.exp_vcf_btn = QPushButton("Export VCF")
        self.exp_bed_btn = QPushButton("Export BED")
        exp_l.addWidget(self.exp_vcf_btn)
        exp_l.addWidget(self.exp_bed_btn)
        t3l.addWidget(exp_w)

        t3l.addStretch()
        self.tabs.addTab(t3, "Analysis")

    def _build_tab_info(self):
        t4 = QWidget()
        t4.setAttribute(Qt.WA_TranslucentBackground, True)
        t4l = QVBoxLayout(t4)
        t4l.setSpacing(5)
        t4l.setContentsMargins(6, 8, 6, 6)

        t4l.addWidget(_lbl("Properties"))
        self.info_edit = QTextEdit()
        self.info_edit.setReadOnly(True)
        self.info_edit.setMinimumHeight(80)
        t4l.addWidget(self.info_edit)

        t4l.addWidget(_lbl("Genes"))
        self.gene_list = QListWidget()
        self.gene_list.setMinimumHeight(80)
        t4l.addWidget(self.gene_list)

        t4l.addWidget(_lbl("Log"))
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setPlainText("[GenomeLab] Initialised\n[GenomeLab] Renderer: ModernGL + QPainter\n")
        t4l.addWidget(self.log_edit)

        t4l.addStretch()
        self.tabs.addTab(t4, "Info")

    def _build_tab_import(self):
        t5 = QWidget()
        t5.setAttribute(Qt.WA_TranslucentBackground, True)
        t5l = QVBoxLayout(t5)
        t5l.setSpacing(5)
        t5l.setContentsMargins(6, 8, 6, 6)

        t5l.addWidget(_lbl("Load Files"))
        self.load_vcf_edit = QLineEdit()
        self.load_vcf_edit.setPlaceholderText("Path to .vcf or .vcf.gz")
        t5l.addWidget(self.load_vcf_edit)
        self.load_vcf_btn = QPushButton("Load VCF")
        t5l.addWidget(self.load_vcf_btn)

        self.load_gff_edit = QLineEdit()
        self.load_gff_edit.setPlaceholderText("Path to .gff3 or .gtf")
        t5l.addWidget(self.load_gff_edit)
        self.load_gff_btn = QPushButton("Load GFF/GTF")
        t5l.addWidget(self.load_gff_btn)

        self.load_fasta_edit = QLineEdit()
        self.load_fasta_edit.setPlaceholderText("Path to .fa or .fa.gz")
        t5l.addWidget(self.load_fasta_edit)
        self.load_fasta_btn = QPushButton("Load FASTA")
        t5l.addWidget(self.load_fasta_btn)

        t5l.addWidget(_lbl("Demo Datasets"))
        self.demo_combo = QComboBox()
        self.demo_combo.addItems(list(DEMOS.keys()))
        t5l.addWidget(self.demo_combo)
        self.load_demo_btn = QPushButton("Load Demo")
        t5l.addWidget(self.load_demo_btn)

        self.status_lbl = QLabel("")
        self.status_lbl.setWordWrap(True)
        self.status_lbl.setStyleSheet("color:#6a7a8a;font-size:10px;background:transparent")
        t5l.addWidget(self.status_lbl)

        t5l.addStretch()
        self.tabs.addTab(t5, "Import")

    # ── Update UI ──

    def _update_ui(self):
        """Sync UI with genome state."""
        genome = self.genome
        viewer = self.viewer

        # Info panel
        info_lines = [f"Source: {genome.info.get('name', genome.info.get('filename', ''))}"]
        info_lines.append(f"Assembly: {genome.info.get('assembly', 'unknown')}")
        info_lines.append(f"Variants: {genome.info.get('n_variants', len(genome.variants))}")
        if genome._filtered is not None:
            info_lines.append(f"Filtered: {len(genome._filtered)}")
        info_lines.append(f"Genes: {len(genome.genes)}")
        info_lines.append(f"View: {viewer.chrom}:{viewer.view_start:,}-{viewer.view_end:,}")
        vt = genome.info.get('variant_types', {})
        if vt:
            info_lines.append("Types: " + ", ".join(f"{k}={v}" for k, v in vt.items()))
        self.info_edit.setPlainText('\n'.join(info_lines))

        # Gene list
        self.gene_list.clear()
        for g in genome.genes:
            self.gene_list.addItem(f"{g['name']:12s}  {g['chrom']}:{g['start']:,}-{g['end']:,}  {g.get('biotype','')}")

        # Variant list (show first 200)
        self.var_list.clear()
        vlist = genome._filtered if genome._filtered is not None else genome.variants
        visible = [v for v in vlist if v['chrom'] == viewer.chrom
                   and viewer.view_start <= v['pos'] <= viewer.view_end][:200]
        for v in visible:
            vstr = f"{v['chrom']}:{v['pos']:>12,}  {v['ref']}>{v['alt']}  Q={v['qual']:.0f}  {v.get('type','.')}  {v.get('impact','.')}  {v.get('gene','.')}"
            self.var_list.addItem(vstr)

    # ── Wrap methods to auto-update UI ──

    def _wrap_methods(self):
        genome = self.genome

        _orig_load_demo = genome.load_demo
        def _load_demo_wrap(name="brca"):
            _orig_load_demo(name); self._update_ui(); return genome
        genome.load_demo = _load_demo_wrap

        _orig_load_vcf = genome.load_vcf
        def _load_vcf_wrap(path):
            _orig_load_vcf(path); self._update_ui(); return genome
        genome.load_vcf = _load_vcf_wrap

        _orig_load_gff = genome.load_gff
        def _load_gff_wrap(path):
            _orig_load_gff(path); self._update_ui(); return genome
        genome.load_gff = _load_gff_wrap

        _orig_load_fasta = genome.load_fasta
        def _load_fasta_wrap(path):
            _orig_load_fasta(path); self._update_ui(); return genome
        genome.load_fasta = _load_fasta_wrap

        _orig_filter = genome.filter_variants
        def _filter_wrap(**kwargs):
            _orig_filter(**kwargs); self._update_ui(); return genome
        genome.filter_variants = _filter_wrap

        _orig_reset = genome.reset_filter
        def _reset_wrap():
            _orig_reset(); self._update_ui(); return genome
        genome.reset_filter = _reset_wrap

        _orig_goto = genome.goto
        def _goto_wrap(region_or_chrom, start=None, end=None):
            _orig_goto(region_or_chrom, start, end); self._update_ui(); return genome
        genome.goto = _goto_wrap

    # ── Wire signals ──

    def _wire_signals(self):
        genome = self.genome
        viewer = self.viewer

        self.chrom_combo.currentTextChanged.connect(lambda ch: genome.set_chromosome(ch))
        self.region_edit.returnPressed.connect(lambda: genome.goto(self.region_edit.text().strip()))
        self.zoom_in_btn.clicked.connect(lambda: (genome.zoom_in(), self._update_ui()))
        self.zoom_out_btn.clicked.connect(lambda: (genome.zoom_out(), self._update_ui()))

        self.show_genes_cb.toggled.connect(lambda c: setattr(viewer, 'show_genes', c))
        self.show_variants_cb.toggled.connect(lambda c: setattr(viewer, 'show_variants', c))
        self.show_cov_cb.toggled.connect(lambda c: setattr(viewer, 'show_coverage', c))
        self.show_seq_cb.toggled.connect(lambda c: genome.show_sequence() if c else setattr(viewer, 'show_sequence', False))
        self.color_combo.currentTextChanged.connect(lambda m: genome.color_by_mode(m))

        self.apply_filter_btn.clicked.connect(self._apply_filter)
        self.reset_filter_btn.clicked.connect(lambda: (genome.reset_filter(), self.status_lbl.setText("\u2713 Filters cleared")))

        self.manhattan_btn.clicked.connect(lambda: genome.overlay_manhattan())
        self.af_btn.clicked.connect(lambda: genome.overlay_frequency())
        self.qual_btn.clicked.connect(lambda: genome.overlay_qual())
        self.karyo_btn.clicked.connect(lambda: genome.ideogram())
        self.clear_ov_btn.clicked.connect(lambda: genome.clear_overlays())

        self.motif_btn.clicked.connect(lambda: genome.find_motif(self.motif_edit.text().strip()) if self.motif_edit.text().strip() else None)

        self.exp_vcf_btn.clicked.connect(lambda: (genome.export_vcf(), self.status_lbl.setText("\u2713 Exported VCF")))
        self.exp_bed_btn.clicked.connect(lambda: (genome.export_bed(), self.status_lbl.setText("\u2713 Exported BED")))

        self.load_vcf_btn.clicked.connect(lambda: genome.load_vcf(self.load_vcf_edit.text().strip()) if self.load_vcf_edit.text().strip() else None)
        self.load_gff_btn.clicked.connect(lambda: genome.load_gff(self.load_gff_edit.text().strip()) if self.load_gff_edit.text().strip() else None)
        self.load_fasta_btn.clicked.connect(lambda: genome.load_fasta(self.load_fasta_edit.text().strip()) if self.load_fasta_edit.text().strip() else None)
        self.load_demo_btn.clicked.connect(lambda: genome.load_demo(self.demo_combo.currentText()))

        self.gene_list.currentRowChanged.connect(lambda r: genome.goto(genome.genes[r]['name']) if 0 <= r < len(genome.genes) else None)

    def _apply_filter(self):
        kwargs = {}
        g = self.filter_gene_edit.text().strip()
        if g: kwargs['gene'] = g
        imp = self.filter_impact_combo.currentText()
        if imp != "All Impacts": kwargs['impact'] = imp
        vt = self.filter_type_combo.currentText()
        if vt != "All Types": kwargs['vtype'] = vt
        qv = self.qual_sl.value()
        if qv > 0: kwargs['min_qual'] = qv
        self.genome.filter_variants(**kwargs)
        self.status_lbl.setText(f"\u2713 Filtered: {len(self.genome._filtered or [])} variants")


# ═══════════════════════════════════════════════════════════════
#  INSTANTIATE
# ═══════════════════════════════════════════════════════════════

genome_app = GenomeLabApp()
genome_widget = genome_app.main_widget
genome = genome_app.genome
gen = genome  # alias
genome_viewer = genome_app.viewer

# ═══════════════════════════════════════════════════════════════
#  ADD TO SCENE
# ═══════════════════════════════════════════════════════════════

genome_proxy = graphics_scene.addWidget(genome_widget)
genome_proxy.setPos(0, 0)
genome_proxy.setFlag(QGraphicsItem.ItemIsMovable, True)
genome_shadow = QGraphicsDropShadowEffect()
genome_shadow.setBlurRadius(60)
genome_shadow.setOffset(45, 45)
genome_shadow.setColor(QColor(0, 0, 0, 120))
genome_proxy.setGraphicsEffect(genome_shadow)
genome_widget.resize(1400, 850)

# Center in current view
_vr = graphics_view.mapToScene(graphics_view.viewport().rect()).boundingRect()
genome_proxy.setPos(_vr.center().x() - genome_widget.width() / 2,
             _vr.center().y() - genome_widget.height() / 2)