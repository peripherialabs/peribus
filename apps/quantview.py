"""
QuantView — Financial Terminal with Native GL Rendering
═══════════════════════════════════════════════════════════════
Offscreen FBO → QImage → QPainter (same pattern as chemlab/car_3_gl)
"""

import math, random, time
import numpy as np
from datetime import datetime, timedelta

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QSlider, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QTabWidget, QTextEdit, QScrollArea,
    QGraphicsItem, QGraphicsDropShadowEffect
)
from PySide6.QtCore import Qt, QTimer, QPointF
from PySide6.QtGui import (
    QPainter, QColor, QFont, QPen, QBrush, QImage,
    QLinearGradient, QPainterPath, QPolygonF
)

import moderngl
import glm

# ═══════════════════════════════════════════════════════════════
#  SHADERS
# ═══════════════════════════════════════════════════════════════

SURFACE_VERT = """
#version 330
uniform mat4 mvp;
uniform mat4 model;
in vec3 in_position;
in vec3 in_color;
in vec3 in_normal;
out vec3 v_color;
out vec3 v_normal;
out vec3 v_pos;
void main() {
    gl_Position = mvp * vec4(in_position, 1.0);
    v_color = in_color;
    v_normal = mat3(model) * in_normal;
    v_pos = (model * vec4(in_position, 1.0)).xyz;
}
"""

SURFACE_FRAG = """
#version 330
uniform vec3 cam_pos;
uniform vec3 fog_color;
in vec3 v_color;
in vec3 v_normal;
in vec3 v_pos;
out vec4 frag;
void main() {
    vec3 N = normalize(v_normal);
    vec3 L = normalize(vec3(0.4, 1.0, 0.6));
    vec3 V = normalize(cam_pos - v_pos);
    vec3 H = normalize(L + V);
    float diff = max(dot(N, L), 0.0) * 0.6 + 0.25;
    float spec = pow(max(dot(N, H), 0.0), 64.0) * 0.4;
    float rim = pow(1.0 - max(dot(N, V), 0.0), 3.0) * 0.25;
    vec3 col = v_color * diff + vec3(1.0) * spec + vec3(0.1, 0.55, 0.45) * rim;
    float dist = length(cam_pos - v_pos);
    float fog = clamp((dist - 2.0) / 8.0, 0.0, 0.35);
    col = mix(col, fog_color, fog);
    frag = vec4(col, 0.92);
}
"""

GRID_VERT = """
#version 330
uniform mat4 mvp;
in vec3 in_position;
in float in_alpha;
out float v_alpha;
void main() { gl_Position = mvp * vec4(in_position, 1.0); v_alpha = in_alpha; }
"""

GRID_FRAG = """
#version 330
in float v_alpha;
out vec4 frag;
void main() { frag = vec4(0.55, 0.62, 0.70, v_alpha * 0.30); }
"""

WIRE_VERT = """
#version 330
uniform mat4 mvp;
in vec3 in_position;
void main() { gl_Position = mvp * vec4(in_position, 1.0); }
"""

WIRE_FRAG = """
#version 330
out vec4 frag;
void main() { frag = vec4(0.40, 0.48, 0.56, 0.18); }
"""


# ═══════════════════════════════════════════════════════════════
#  PALETTE + STYLESHEET  (Light Mode)
# ═══════════════════════════════════════════════════════════════

BG_LIGHT = "#f5f6f8"
BG_PANEL = "#ffffff"
BG_INPUT = "#eef0f4"
BG_HOVER = "#e4e7ed"
BORDER = "#d0d5dd"
TEXT = "#2c3e50"
TEXT_DIM = "#5a6a7a"
TEXT_MUTE = "#8896a4"
ACCENT = "#0091ff"
ACCENT_DIM = "#d6ecff"

STYLESHEET = f"""
QWidget {{ background-color:{BG_PANEL}; color:{TEXT}; font-family:'Consolas','SF Mono','Menlo',monospace; font-size:11px; }}
QPushButton {{ background:{BG_INPUT}; border:1px solid {BORDER}; border-radius:5px; padding:7px 12px; color:{ACCENT}; font-weight:bold; font-size:10px; }}
QPushButton:hover {{ background:{BG_HOVER}; border-color:{ACCENT}; }}
QSlider::groove:horizontal {{ height:4px; background:{BORDER}; border-radius:2px; }}
QSlider::handle:horizontal {{ background:{ACCENT}; width:14px; margin:-5px 0; border-radius:7px; }}
QComboBox {{ background:{BG_INPUT}; border:1px solid {BORDER}; border-radius:4px; padding:5px 8px; color:{TEXT}; }}
QComboBox QAbstractItemView {{ background:{BG_PANEL}; border:1px solid {BORDER}; color:{TEXT}; selection-background-color:{BG_HOVER}; }}
QTextEdit {{ background:{BG_LIGHT}; border:1px solid {BORDER}; border-radius:4px; font-family:'Consolas',monospace; font-size:10px; color:{TEXT_DIM}; padding:6px; }}
QCheckBox {{ spacing:8px; color:{TEXT_DIM}; font-size:11px; }}
QCheckBox::indicator {{ width:16px; height:16px; border-radius:3px; border:1px solid {BORDER}; background:{BG_INPUT}; }}
QCheckBox::indicator:checked {{ background:{ACCENT_DIM}; border-color:{ACCENT}; }}
QTabWidget::pane {{ border:1px solid {BORDER}; background:{BG_PANEL}; border-top:none; }}
QTabBar::tab {{ background:{BG_LIGHT}; color:{TEXT_DIM}; padding:8px 14px; font-size:10px; font-weight:bold; border:1px solid {BORDER}; border-bottom:none; margin-right:1px; }}
QTabBar::tab:selected {{ background:{BG_PANEL}; color:{ACCENT}; border-bottom:2px solid {ACCENT}; }}
QLabel {{ background:transparent; }}
QScrollArea {{ border:none; background:transparent; }}
QDoubleSpinBox,QSpinBox {{ background:{BG_INPUT}; border:1px solid {BORDER}; border-radius:4px; padding:4px 6px; color:{TEXT}; font-size:11px; }}
"""


# ═══════════════════════════════════════════════════════════════
#  MARKET DATA + INDICATORS + BLACK-SCHOLES
# ═══════════════════════════════════════════════════════════════

def generate_ohlcv(ticker, days=252, base_price=100.0, volatility=0.02, drift=0.0003, seed=None):
    if seed is not None: random.seed(seed)
    data, price = [], base_price
    start = datetime.now() - timedelta(days=days)
    for i in range(days):
        dt = start + timedelta(days=i)
        if dt.weekday() >= 5: continue
        price *= math.exp(drift + volatility * random.gauss(0, 1))
        o = price * (1 + random.gauss(0, volatility * 0.3)); c = price
        h = max(o, c) * (1 + abs(random.gauss(0, volatility * 0.6)))
        l = min(o, c) * (1 - abs(random.gauss(0, volatility * 0.6)))
        data.append({'date': dt.strftime('%Y-%m-%d'), 'open': round(o,2), 'high': round(h,2), 'low': round(l,2), 'close': round(c,2), 'volume': max(100000, int(random.gauss(5e6, 2e6)))})
    return data

TICKERS = {
    "AAPL":{"name":"Apple Inc.","base":185.0,"vol":0.018,"drift":0.0004,"seed":42,"sector":"Technology"},
    "TSLA":{"name":"Tesla Inc.","base":240.0,"vol":0.035,"drift":0.0002,"seed":77,"sector":"Automotive"},
    "NVDA":{"name":"NVIDIA Corp.","base":480.0,"vol":0.028,"drift":0.0008,"seed":101,"sector":"Semiconductors"},
    "JPM":{"name":"JPMorgan Chase","base":155.0,"vol":0.015,"drift":0.0003,"seed":55,"sector":"Financials"},
    "AMZN":{"name":"Amazon.com","base":145.0,"vol":0.022,"drift":0.0005,"seed":33,"sector":"E-Commerce"},
    "MSFT":{"name":"Microsoft Corp.","base":370.0,"vol":0.016,"drift":0.0004,"seed":88,"sector":"Technology"},
    "SPY":{"name":"S&P 500 ETF","base":450.0,"vol":0.012,"drift":0.0003,"seed":10,"sector":"Index"},
}

def compute_sma(c, p):
    s=[None]*len(c)
    for i in range(p-1,len(c)): s[i]=sum(c[i-p+1:i+1])/p
    return s

def compute_ema(c, p):
    e=[None]*len(c); k=2.0/(p+1); e[p-1]=sum(c[:p])/p
    for i in range(p,len(c)): e[i]=c[i]*k+e[i-1]*(1-k)
    return e

def compute_bollinger(c, p=20, ns=2):
    s=compute_sma(c,p); u=[None]*len(c); l=[None]*len(c)
    for i in range(p-1,len(c)):
        w=c[i-p+1:i+1]; sd=math.sqrt(sum((x-s[i])**2 for x in w)/p)
        u[i]=s[i]+ns*sd; l[i]=s[i]-ns*sd
    return u,s,l

def norm_cdf(x):
    a1,a2,a3,a4,a5=0.254829592,-0.284496736,1.421413741,-1.453152027,1.061405429
    p=0.3275911; sgn=1 if x>=0 else -1; x=abs(x)/math.sqrt(2)
    t=1.0/(1.0+p*x); y=1.0-(((((a5*t+a4)*t)+a3)*t+a2)*t+a1)*t*math.exp(-x*x)
    return 0.5*(1.0+sgn*y)

def norm_pdf(x): return math.exp(-0.5*x*x)/math.sqrt(2*math.pi)

def black_scholes(S,K,T,r,sigma,otype='call'):
    if T<=0 or sigma<=0:
        intr=max(S-K,0) if otype=='call' else max(K-S,0)
        return {'price':intr,'delta':1 if S>K else 0,'gamma':0,'theta':0,'vega':0,'rho':0}
    d1=(math.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*math.sqrt(T)); d2=d1-sigma*math.sqrt(T)
    if otype=='call':
        price=S*norm_cdf(d1)-K*math.exp(-r*T)*norm_cdf(d2); delta=norm_cdf(d1); rho=K*T*math.exp(-r*T)*norm_cdf(d2)/100
    else:
        price=K*math.exp(-r*T)*norm_cdf(-d2)-S*norm_cdf(-d1); delta=norm_cdf(d1)-1; rho=-K*T*math.exp(-r*T)*norm_cdf(-d2)/100
    gamma=norm_pdf(d1)/(S*sigma*math.sqrt(T))
    theta=(-(S*norm_pdf(d1)*sigma)/(2*math.sqrt(T))-r*K*math.exp(-r*T)*norm_cdf(d2 if otype=='call' else -d2)*(1 if otype=='call' else -1))/365
    vega=S*norm_pdf(d1)*math.sqrt(T)/100
    return {k:round(v,6) for k,v in {'price':price,'delta':delta,'gamma':gamma,'theta':theta,'vega':vega,'rho':rho}.items()}

def generate_vol_surface(S, r=0.05, base_vol=0.25):
    strikes=[S*(0.7+i*0.05) for i in range(13)]; expiries=[i/12.0 for i in range(1,13)]
    surface=[]
    for T in expiries:
        row=[]
        for K in strikes:
            m=math.log(K/S)
            row.append(round(max(0.05, base_vol+0.15*m**2+0.02*abs(m)+0.03*math.sqrt(T)-0.05*m+random.gauss(0,0.005)),4))
        surface.append(row)
    return {'strikes':[round(k,2) for k in strikes],'expiries':[round(t,4) for t in expiries],'ivs':surface}

SAMPLE_PORTFOLIO=[
    {"ticker":"AAPL","shares":50,"avg_cost":172.50},{"ticker":"NVDA","shares":30,"avg_cost":420.00},
    {"ticker":"MSFT","shares":25,"avg_cost":355.00},{"ticker":"JPM","shares":40,"avg_cost":148.00},
    {"ticker":"AMZN","shares":35,"avg_cost":132.00},
]

def compute_portfolio_stats(portfolio, td):
    tv=0; tc=0; pos=[]
    for p_ in portfolio:
        t=p_['ticker']
        if t in td and td[t]:
            cur=td[t][-1]['close']; val=p_['shares']*cur; cost=p_['shares']*p_['avg_cost']
            pnl=val-cost; tv+=val; tc+=cost
            pos.append({'ticker':t,'shares':p_['shares'],'avg_cost':p_['avg_cost'],'current':cur,'value':round(val,2),'pnl':round(pnl,2),'pnl_pct':round((pnl/cost)*100,2) if cost else 0})
    for p_ in pos: p_['weight']=round((p_['value']/tv)*100,2) if tv else 0
    return {'positions':pos,'total_value':round(tv,2),'total_cost':round(tc,2),'total_pnl':round(tv-tc,2),'total_pnl_pct':round(((tv-tc)/tc)*100,2) if tc else 0}


# ═══════════════════════════════════════════════════════════════
#  3D VIEWPORT — Offscreen FBO → QImage → QPainter
# ═══════════════════════════════════════════════════════════════

class FinanceViewport(QWidget):
    """Offscreen ModernGL rendering (standalone context) blitted to QWidget."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(600, 400)
        self.setAttribute(Qt.WA_OpaquePaintEvent)

        self.view_mode = 'chart'
        self.chart_data = []; self.overlays = {}
        self.ticker_label = ""; self.ticker_name = ""
        self.vol_surface = None; self.portfolio_stats = None; self.greeks_data = None

        # 3D camera
        self.rot_x = 0.45; self.rot_y = 0.7; self.cam_dist = 3.5
        self.dragging = False; self.last_mouse = QPointF(0, 0)
        self.auto_rot = 0.0; self.hover_idx = -1

        # GL state — lazy init
        self._gl_ready = False; self.ctx = None; self.fbo = None
        self._fbo_w = 0; self._fbo_h = 0; self._frame = None
        self.prog_surf = None; self.prog_grid = None; self.prog_wire = None
        self.vao_surf = None; self.vao_grid = None; self.vao_wire = None
        self.surf_idx_count = 0

        self._timer = QTimer(); self._timer.timeout.connect(self._tick); self._timer.start(30)

    # ── GL lazy init (standalone offscreen) ───────────────────
    def _ensure_gl(self):
        if self._gl_ready: return
        try:
            self.ctx = moderngl.create_context(standalone=True)
            self.ctx.enable(moderngl.DEPTH_TEST)
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            self.prog_surf = self.ctx.program(vertex_shader=SURFACE_VERT, fragment_shader=SURFACE_FRAG)
            self.prog_grid = self.ctx.program(vertex_shader=GRID_VERT, fragment_shader=GRID_FRAG)
            self.prog_wire = self.ctx.program(vertex_shader=WIRE_VERT, fragment_shader=WIRE_FRAG)
            self._build_grid()
            self._resize_fbo(max(self.width(), 320), max(self.height(), 200))
            self._gl_ready = True
        except Exception as e:
            print(f"[QuantView] GL init failed: {e}")
            self._gl_ready = True

    def _resize_fbo(self, w, h):
        if w == self._fbo_w and h == self._fbo_h and self.fbo: return
        if self.fbo: self.fbo.release()
        self._fbo_w = w; self._fbo_h = h
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((w, h), 4)],
            depth_attachment=self.ctx.depth_renderbuffer((w, h)))

    def _build_grid(self):
        verts = []
        n = 16; sz = 1.2; step = sz * 2 / n
        for i in range(n + 1):
            t = -sz + i * step; fade = 1.0 - abs(t) / sz * 0.6
            verts.extend([t, -1.2, -sz, fade, t, -1.2, sz, fade])
            verts.extend([-sz, -1.2, t, fade, sz, -1.2, t, fade])
        data = np.array(verts, dtype='f4')
        vbo = self.ctx.buffer(data.tobytes())
        self.vao_grid = self.ctx.vertex_array(self.prog_grid, [(vbo, '3f 1f', 'in_position', 'in_alpha')])
        self.grid_vert_count = len(verts) // 4

    def upload_surface(self):
        """Build GPU buffers from vol_surface data."""
        if not self.vol_surface or not self.ctx: return
        s = self.vol_surface
        strikes = s['strikes']; expiries = s['expiries']; ivs = s['ivs']
        nS = len(strikes); nT = len(expiries)
        sMin, sMax = strikes[0], strikes[-1]; sRange = max(sMax - sMin, 1)
        tMax = max(expiries[-1], 0.01)

        verts = []
        for ti in range(nT):
            for si in range(nS):
                x = (strikes[si] - sMin) / sRange * 2.0 - 1.0
                y = ivs[ti][si] * 4.0 - 1.0
                z = expiries[ti] / tMax * 2.0 - 1.0
                t = min(1.0, max(0.0, (ivs[ti][si] - 0.08) / 0.45))
                r = t * 0.95; g = 0.83 - t * 0.45; b = 0.67 - t * 0.35
                verts.append((x, y, z, r, g, b))

        normals = [[0, 0, 0] for _ in range(nT * nS)]
        indices = []
        for ti in range(nT - 1):
            for si in range(nS - 1):
                a = ti * nS + si; b = a + 1; c = (ti + 1) * nS + si; d = c + 1
                indices.extend([a, b, c, b, d, c])
                p0 = np.array(verts[a][:3]); p1 = np.array(verts[b][:3]); p2 = np.array(verts[c][:3])
                n1 = np.cross(p1 - p0, p2 - p0)
                for idx in [a, b, c]: normals[idx] = [normals[idx][j] + n1[j] for j in range(3)]
                p3 = np.array(verts[d][:3]); n2 = np.cross(p3 - p1, p2 - p1)
                for idx in [b, d, c]: normals[idx] = [normals[idx][j] + n2[j] for j in range(3)]

        final = []
        for i, v in enumerate(verts):
            nx, ny, nz = normals[i]; l = math.sqrt(nx * nx + ny * ny + nz * nz)
            if l > 0: nx /= l; ny /= l; nz /= l
            else: ny = 1.0
            final.extend([v[0], v[1], v[2], v[3], v[4], v[5], nx, ny, nz])

        vbo = self.ctx.buffer(np.array(final, dtype='f4').tobytes())
        ibo = self.ctx.buffer(np.array(indices, dtype='i4').tobytes())
        if self.vao_surf: self.vao_surf.release()
        self.vao_surf = self.ctx.vertex_array(self.prog_surf, [(vbo, '3f 3f 3f', 'in_position', 'in_color', 'in_normal')], index_buffer=ibo)
        self.surf_idx_count = len(indices)

        wire_data = np.array([c for v in verts for c in v[:3]], dtype='f4')
        wire_vbo = self.ctx.buffer(wire_data.tobytes())
        if self.vao_wire: self.vao_wire.release()
        self.vao_wire = self.ctx.vertex_array(self.prog_wire, [(wire_vbo, '3f', 'in_position')], index_buffer=ibo)

    # ── Render 3D to FBO ──────────────────────────────────────
    def _render_surface(self):
        if not self._gl_ready or not self.vao_surf: return
        w, h = max(self.width(), 320), max(self.height(), 200)
        self._resize_fbo(w, h)
        self.fbo.use(); self.ctx.viewport = (0, 0, w, h)
        self.ctx.clear(0.94, 0.95, 0.97, 1.0)

        aspect = w / h; ay = self.rot_y + self.auto_rot
        cx = self.cam_dist * math.sin(ay) * math.cos(self.rot_x)
        cy = self.cam_dist * math.sin(self.rot_x) + 0.5
        cz = self.cam_dist * math.cos(ay) * math.cos(self.rot_x)
        proj = glm.perspective(glm.radians(42.0), float(aspect), 0.1, 100.0)
        view = glm.lookAt(glm.vec3(float(cx), float(cy), float(cz)), glm.vec3(0.0, -0.2, 0.0), glm.vec3(0.0, 1.0, 0.0))
        model = glm.mat4(1.0); mvp = proj * view * model

        self.prog_grid['mvp'].write(mvp)
        self.vao_grid.render(moderngl.LINES)

        self.prog_surf['mvp'].write(mvp)
        self.prog_surf['model'].write(model)
        self.prog_surf['cam_pos'].write(glm.vec3(float(cx), float(cy), float(cz)))
        self.prog_surf['fog_color'].write(glm.vec3(0.94, 0.95, 0.97))
        self.vao_surf.render(moderngl.TRIANGLES)

        self.ctx.wireframe = True
        self.prog_wire['mvp'].write(mvp)
        self.vao_wire.render(moderngl.TRIANGLES)
        self.ctx.wireframe = False

        raw = self.fbo.color_attachments[0].read()
        self._frame = QImage(raw, w, h, w * 4, QImage.Format_RGBA8888).mirrored(False, True)

    # ── paintEvent — draw cached frame + 2D overlays ─────────
    def paintEvent(self, event):
        self._ensure_gl()
        p = QPainter(self)
        try:
            p.setRenderHint(QPainter.Antialiasing); p.setRenderHint(QPainter.TextAntialiasing)
            w, h = self.width(), self.height()

            if self.view_mode == 'surface':
                try:
                    self._render_surface()
                except Exception as e:
                    print(f"[QuantView] Surface render error: {e}")
                if self._frame and not self._frame.isNull(): p.drawImage(0, 0, self._frame)
                else: p.fillRect(0, 0, w, h, QColor(240, 242, 245))
                self._paint_surface_hud(p, w, h)
            else:
                bg = QLinearGradient(0, 0, 0, h)
                bg.setColorAt(0, QColor(245, 246, 248)); bg.setColorAt(1, QColor(235, 237, 242))
                p.fillRect(0, 0, w, h, bg)
                if self.view_mode == 'chart': self._paint_chart(p, w, h)
                elif self.view_mode == 'portfolio': self._paint_portfolio(p, w, h)
                elif self.view_mode == 'greeks': self._paint_greeks(p, w, h)
        except Exception as e:
            print(f"[QuantView] Paint error: {e}")
        finally:
            p.end()

    # ── Shadow box helper (light mode) ────────────────────────
    def _sbox(self, p, x, y, w, h):
        for i in range(8, 0, -1): p.fillRect(int(x+i), int(y+i), int(w), int(h), QColor(0,0,0,max(1,6-i)))
        p.fillRect(int(x), int(y), int(w), int(h), QColor(255,255,255,245))
        p.setPen(QPen(QColor(208,213,221),1)); p.drawRect(int(x), int(y), int(w), int(h))

    # ── CHART ─────────────────────────────────────────────────
    def _paint_chart(self, p, w, h):
        if not self.chart_data:
            p.setFont(QFont("Consolas",11)); p.setPen(QColor(136,150,168)); p.drawText(w//2-100,h//2,"Select a ticker"); return
        data=self.chart_data; n=len(data); mg={'l':60,'r':70,'t':80,'b':130}
        cw=w-mg['l']-mg['r']; vol_h=80; ch=max(50,h-mg['t']-mg['b']-vol_h-10)
        gap=cw/n if n else 1; bw=max(2,gap*0.7)
        pmin=min(d['low'] for d in data)*0.998; pmax=max(d['high'] for d in data)*1.002; pr=pmax-pmin if pmax!=pmin else 1
        def xf(i): return mg['l']+i*gap+gap/2
        def yf(v): return mg['t']+ch*(1-(v-pmin)/pr)

        # Grid
        p.setPen(QPen(QColor(220,224,232),0.5))
        for i in range(9):
            gy=mg['t']+ch*i/8; p.drawLine(int(mg['l']),int(gy),int(w-mg['r']),int(gy))
            pv=pmax-(pmax-pmin)*i/8; p.setFont(QFont("Consolas",8)); p.setPen(QPen(QColor(120,135,150),1))
            p.drawText(int(w-mg['r']+6),int(gy+4),f"{pv:.2f}"); p.setPen(QPen(QColor(220,224,232),0.5))
        step=max(1,n//10); p.setFont(QFont("Consolas",7)); p.setPen(QColor(136,150,168))
        for i in range(0,n,step): p.drawText(int(xf(i)-16),int(mg['t']+ch+16),data[i]['date'][5:])

        # Area fill
        area=QPainterPath(); area.moveTo(xf(0),yf(data[0]['close']))
        for i in range(1,n): area.lineTo(xf(i),yf(data[i]['close']))
        area.lineTo(xf(n-1),mg['t']+ch); area.lineTo(xf(0),mg['t']+ch); area.closeSubpath()
        up=data[-1]['close']>=data[0]['close']
        gr=QLinearGradient(0,mg['t'],0,mg['t']+ch)
        gr.setColorAt(0,QColor(0,145,255,40) if up else QColor(235,68,90,40))
        gr.setColorAt(1,QColor(0,145,255,0) if up else QColor(235,68,90,0))
        p.fillPath(area,gr)

        # Overlays
        oc={'sma20':QColor(230,140,0,200),'sma50':QColor(210,90,0,200),'ema12':QColor(40,130,220,200),
            'bb_upper':QColor(80,140,220,100),'bb_mid':QColor(80,140,220,140),'bb_lower':QColor(80,140,220,100)}
        for key,vals in self.overlays.items():
            col=oc.get(key,QColor(136,136,136,150)); p.setPen(QPen(col,1.3)); prev=None
            for i,v in enumerate(vals):
                if v is None: prev=None; continue
                pt=QPointF(xf(i),yf(v))
                if prev: p.drawLine(prev,pt)
                prev=pt
        if 'bb_upper' in self.overlays and 'bb_lower' in self.overlays:
            bb=QPainterPath(); started=False; ui=[]
            for i,v in enumerate(self.overlays['bb_upper']):
                if v is None: continue
                pt=QPointF(xf(i),yf(v))
                if not started: bb.moveTo(pt); started=True
                else: bb.lineTo(pt)
                ui.append(i)
            for i in reversed(ui):
                v=self.overlays['bb_lower'][i]
                if v is not None: bb.lineTo(QPointF(xf(i),yf(v)))
            bb.closeSubpath(); p.fillPath(bb,QColor(80,140,220,18))

        # Candlesticks
        for i,bar in enumerate(data):
            x=xf(i); up_=bar['close']>=bar['open']; col=QColor(0,145,255) if up_ else QColor(235,68,90)
            p.setPen(QPen(col,1)); p.drawLine(int(x),int(yf(bar['high'])),int(x),int(yf(bar['low'])))
            oy=yf(bar['open']); cy=yf(bar['close']); bt=min(oy,cy); bh=max(abs(cy-oy),1)
            if bw>3: p.fillRect(int(x-bw/2-2),int(bt-2),int(bw+4),int(bh+4),QColor(0,145,255,25) if up_ else QColor(235,68,90,25))
            p.fillRect(int(x-bw/2),int(bt),int(bw),int(bh),QColor(0,145,255,230) if up_ else QColor(235,68,90,230))
            if i==self.hover_idx: p.fillRect(int(x-bw/2-1),int(bt-1),int(bw+2),int(bh+2),QColor(0,0,0,18))

        # Volume
        vt=mg['t']+ch+20; mv=max(d['volume'] for d in data)
        for i,bar in enumerate(data):
            x=xf(i); up_=bar['close']>=bar['open']; vh=(bar['volume']/mv)*vol_h*0.8
            p.fillRect(int(x-bw/2),int(vt+vol_h-vh),int(bw),int(vh),QColor(0,145,255,70) if up_ else QColor(235,68,90,70))

        # HUD
        last=data[-1]; prev=data[-2] if len(data)>1 else last
        chg=last['close']-prev['close']; pct=(chg/prev['close'])*100 if prev['close'] else 0
        self._sbox(p,10,10,210,105)
        p.setFont(QFont("Consolas",12,QFont.Bold)); p.setPen(QColor(44,62,80)); p.drawText(20,32,self.ticker_label)
        p.setFont(QFont("Consolas",8)); p.setPen(QColor(136,150,168)); p.drawText(20,46,self.ticker_name)
        u=chg>=0; p.setFont(QFont("Consolas",18,QFont.Bold)); p.setPen(QColor(0,145,255) if u else QColor(235,68,90))
        p.drawText(20,72,f"{last['close']:.2f}")
        p.setFont(QFont("Consolas",9,QFont.Bold)); s='+' if chg>=0 else ''; p.drawText(20,88,f"{s}{chg:.2f} ({s}{pct:.2f}%)")
        p.setFont(QFont("Consolas",7)); p.setPen(QColor(136,150,168))
        p.drawText(20,104,f"O {last['open']:.2f}  H {last['high']:.2f}  L {last['low']:.2f}  V {last['volume']/1e6:.1f}M")
        if 0<=self.hover_idx<n:
            bar=data[self.hover_idx]; bx=w-185; self._sbox(p,bx,48,172,92)
            p.setFont(QFont("Consolas",9,QFont.Bold)); p.setPen(QColor(44,62,80)); p.drawText(bx+10,66,bar['date'])
            p.setFont(QFont("Consolas",8)); p.setPen(QColor(90,110,130))
            p.drawText(bx+10,82,f"O:{bar['open']:.2f}"); p.drawText(bx+10,94,f"H:{bar['high']:.2f}")
            p.drawText(bx+10,106,f"L:{bar['low']:.2f}"); p.drawText(bx+10,118,f"C:{bar['close']:.2f}  V:{bar['volume']/1e6:.1f}M")

    def _paint_surface_hud(self, p, w, h):
        p.setFont(QFont("Consolas",11,QFont.Bold)); p.setPen(QColor(0,145,255)); p.drawText(16,24,"IMPLIED VOLATILITY SURFACE")
        p.setFont(QFont("Consolas",8)); p.setPen(QColor(136,150,168)); p.drawText(16,40,"Strike → | Expiry ↗ | IV ↑")
        sx=w-28
        for i in range(100):
            t=i/100.0; r=int(min(255,t*0.95*255)); g=int(max(0,(0.83-t*0.45)*255)); b=int(max(0,(0.67-t*0.35)*255))
            p.fillRect(sx,60+i*2,14,2,QColor(r,g,b))
        p.setFont(QFont("Consolas",7)); p.setPen(QColor(136,150,168)); p.drawText(sx-28,58,"Low IV"); p.drawText(sx-30,268,"High IV")
        p.drawText(w//2-80,h-12,"Drag to rotate · Scroll to zoom")

    def _paint_portfolio(self, p, w, h):
        if not self.portfolio_stats:
            p.setFont(QFont("Consolas",11)); p.setPen(QColor(136,150,168)); p.drawText(w//2-130,h//2,"Load a portfolio"); return
        st=self.portfolio_stats; pos=st['positions']
        self._sbox(p,30,30,w-60,80)
        p.setFont(QFont("Consolas",8)); p.setPen(QColor(136,150,168)); p.drawText(46,50,"PORTFOLIO VALUE")
        p.setFont(QFont("Consolas",22,QFont.Bold)); p.setPen(QColor(44,62,80)); p.drawText(46,80,f"${st['total_value']:,.2f}")
        u=st['total_pnl']>=0; p.setFont(QFont("Consolas",11,QFont.Bold)); p.setPen(QColor(0,145,255) if u else QColor(235,68,90))
        s='+' if u else ''; p.drawText(46,98,f"{s}${st['total_pnl']:,.2f} ({s}{st['total_pnl_pct']:.2f}%)")
        dc=[QColor(0,145,255),QColor(40,180,120),QColor(230,140,0),QColor(150,90,220),QColor(235,68,90)]
        dcx=w-140; dcy=70; dr=32; sa=90*16
        for i,ps in enumerate(pos):
            sp=int(ps['weight']/100*360*16); p.setPen(Qt.NoPen); p.setBrush(dc[i%len(dc)]); p.drawPie(dcx-dr,dcy-dr,dr*2,dr*2,sa,sp); sa+=sp
        p.setBrush(QColor(255,255,255)); p.drawEllipse(dcx-18,dcy-18,36,36)
        ty=134; hdrs=["TICKER","SHARES","AVG COST","CURRENT","VALUE","P&L","P&L %","WEIGHT"]
        cx_=[30,110,175,260,350,450,540,610]
        p.setFont(QFont("Consolas",8,QFont.Bold)); p.setPen(QColor(136,150,168))
        for j,hd in enumerate(hdrs): p.drawText(cx_[j],ty,hd)
        p.setPen(QPen(QColor(208,213,221),1)); p.drawLine(30,ty+6,w-30,ty+6)
        for i,ps in enumerate(pos):
            ry=ty+22+i*26
            if i%2==0: p.fillRect(26,ry-12,w-52,24,QColor(0,0,0,8))
            p.setBrush(dc[i%len(dc)]); p.setPen(Qt.NoPen); p.drawEllipse(cx_[0],ry-4,6,6)
            p.setFont(QFont("Consolas",10,QFont.Bold)); p.setPen(QColor(44,62,80)); p.drawText(cx_[0]+12,ry+1,ps['ticker'])
            p.setFont(QFont("Consolas",9)); p.setPen(QColor(70,85,100))
            p.drawText(cx_[1],ry+1,str(ps['shares'])); p.drawText(cx_[2],ry+1,f"${ps['avg_cost']:.2f}")
            p.drawText(cx_[3],ry+1,f"${ps['current']:.2f}"); p.drawText(cx_[4],ry+1,f"${ps['value']:,.2f}")
            pp=ps['pnl']>=0; sg='+' if pp else ''; p.setPen(QColor(0,145,255) if pp else QColor(235,68,90))
            p.drawText(cx_[5],ry+1,f"{sg}${ps['pnl']:,.2f}"); p.drawText(cx_[6],ry+1,f"{sg}{ps['pnl_pct']:.1f}%")
            p.setPen(QColor(100,115,130)); p.drawText(cx_[7],ry+1,f"{ps['weight']:.1f}%")

    def _paint_greeks(self, p, w, h):
        if not self.greeks_data:
            p.setFont(QFont("Consolas",11)); p.setPen(QColor(136,150,168)); p.drawText(w//2-120,h//2,"Price an option"); return
        g=self.greeks_data['main']; ladder=self.greeks_data.get('ladder',[]); spot=self.greeks_data.get('spot',0)
        cards=[("PRICE",f"${g['price']:.2f}",f"{g['type']} K={g['strike']:.0f}"),("DELTA Δ",f"{g['delta']:.4f}","Sensitivity"),
               ("GAMMA Γ",f"{g['gamma']:.6f}","Acceleration"),("THETA Θ",f"{g['theta']:.4f}","Time decay"),
               ("VEGA ν",f"{g['vega']:.4f}","Vol sens."),("RHO ρ",f"{g['rho']:.4f}","Rate sens.")]
        cw_=min(180,(w-80)//3); ch_=80
        for i,(lb,val,sub) in enumerate(cards):
            col=i%3; row=i//3; cx_=30+col*(cw_+12); cy_=30+row*(ch_+12)
            self._sbox(p,cx_,cy_,cw_,ch_)
            p.setFont(QFont("Consolas",7,QFont.Bold)); p.setPen(QColor(136,150,168)); p.drawText(cx_+12,cy_+18,lb)
            p.setFont(QFont("Consolas",16,QFont.Bold)); p.setPen(QColor(0,145,255)); p.drawText(cx_+12,cy_+46,val)
            p.setFont(QFont("Consolas",7)); p.setPen(QColor(136,150,168)); p.drawText(cx_+12,cy_+64,sub)
        ty=30+2*(ch_+12)+20; hds=["STRIKE","CALL Δ","CALL $","IV","PUT $","PUT Δ"]; cx__=[30,120,220,320,420,520]
        p.setFont(QFont("Consolas",8,QFont.Bold)); p.setPen(QColor(136,150,168))
        for j,hd in enumerate(hds): p.drawText(cx__[j],ty,hd)
        p.setPen(QPen(QColor(208,213,221),1)); p.drawLine(30,ty+6,w-30,ty+6)
        for i,row in enumerate(ladder):
            ry=ty+22+i*22
            if ry>h-20: break
            if abs(row['strike']-spot)<spot*0.03: p.fillRect(26,ry-12,w-52,20,QColor(0,145,255,15))
            if i%2==0: p.fillRect(26,ry-12,w-52,20,QColor(0,0,0,8))
            p.setFont(QFont("Consolas",9)); itm=row['strike']<=spot
            p.setPen(QColor(0,145,255) if itm else QColor(70,85,100)); p.drawText(cx__[0],ry,f"{row['strike']:.2f}")
            p.setPen(QColor(70,85,100)); p.drawText(cx__[1],ry,f"{row['call_delta']:.3f}"); p.drawText(cx__[2],ry,f"${row['call_price']:.2f}")
            p.drawText(cx__[3],ry,f"{row['iv']*100:.1f}%"); p.drawText(cx__[4],ry,f"${row['put_price']:.2f}"); p.drawText(cx__[5],ry,f"{row['put_delta']:.3f}")

    # ── Mouse ─────────────────────────────────────────────────
    def mousePressEvent(self, e): self.dragging=True; self.last_mouse=e.position()
    def mouseReleaseEvent(self, e): self.dragging=False
    def mouseMoveEvent(self, e):
        pos=e.position()
        if self.dragging and self.view_mode=='surface':
            self.rot_y+=(pos.x()-self.last_mouse.x())*0.008; self.rot_x+=(pos.y()-self.last_mouse.y())*0.008
            self.rot_x=max(-1.2,min(1.2,self.rot_x)); self.last_mouse=pos
        elif self.view_mode=='chart' and self.chart_data:
            n=len(self.chart_data); cw=self.width()-130; gap=cw/n if n else 1
            idx=int((pos.x()-60)/gap); self.hover_idx=idx if 0<=idx<n else -1
        self.update()
    def wheelEvent(self, e):
        if self.view_mode=='surface': self.cam_dist-=e.angleDelta().y()*0.002; self.cam_dist=max(1.5,min(8.0,self.cam_dist)); self.update()
    def leaveEvent(self, e): self.hover_idx=-1; self.update()
    def _tick(self):
        if self.view_mode=='surface' and not self.dragging: self.auto_rot+=0.005
        self.update()


# ═══════════════════════════════════════════════════════════════
#  QUANTVIEW APP CLASS
# ═══════════════════════════════════════════════════════════════

class QuantViewApp:
    """Encapsulates the entire QuantView UI, state, and logic."""

    PERIOD_DAYS = {"1 Month":30, "3 Months":90, "6 Months":180, "1 Year":365, "2 Years":730}

    def __init__(self):
        self.current_data = []
        self.current_ticker = "AAPL"
        self.all_td = {}

        self._build_ui()
        self._connect_signals()

        QTimer.singleShot(100, lambda: self.load_ticker(self.ticker_combo.currentText()))

    # ── UI Construction ───────────────────────────────────────
    @staticmethod
    def _make_label(t):
        l = QLabel(t.upper())
        l.setStyleSheet(f"font-size:9px; letter-spacing:2px; color:{TEXT_MUTE}; font-weight:bold; padding:2px 0; background:transparent;")
        return l

    def _build_ui(self):
        self.main_widget = QWidget()
        self.main_layout = QHBoxLayout(self.main_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # Control panel
        cp = QWidget(); cp.setFixedWidth(310); cp.setStyleSheet(STYLESHEET)
        cs = QScrollArea(); cs.setWidgetResizable(True); cs.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        cs.setStyleSheet(f"QScrollArea{{border:none;background:{BG_PANEL};}} QScrollBar:vertical{{width:6px;background:{BG_LIGHT};}} QScrollBar::handle:vertical{{background:{BORDER};border-radius:3px;min-height:30px;}}")
        ci = QWidget(); cl = QVBoxLayout(ci); cl.setSpacing(4); cl.setContentsMargins(12, 12, 12, 12)

        # Header
        hw = QWidget(); hl = QHBoxLayout(hw); hl.setContentsMargins(0, 0, 0, 8)
        ti_ = QLabel("◈"); ti_.setStyleSheet(f"font-size:22px;background:{BG_INPUT};border:1px solid {BORDER};border-radius:8px;padding:4px 8px;color:{ACCENT};")
        tw = QWidget(); tl_ = QVBoxLayout(tw); tl_.setContentsMargins(8, 0, 0, 0); tl_.setSpacing(0)
        tn_ = QLabel("QuantView"); tn_.setStyleSheet("font-size:15px;font-weight:bold;color:#2c3e50;background:transparent;")
        ts_ = QLabel("OFFSCREEN GL TERMINAL"); ts_.setStyleSheet(f"font-size:8px;letter-spacing:2px;color:{TEXT_MUTE};background:transparent;")
        tl_.addWidget(tn_); tl_.addWidget(ts_); hl.addWidget(ti_); hl.addWidget(tw); hl.addStretch(); cl.addWidget(hw)

        # View buttons
        cl.addWidget(self._make_label("View"))
        vg = QWidget(); vlb = QHBoxLayout(vg); vlb.setContentsMargins(0, 0, 0, 0); vlb.setSpacing(3)
        self.view_btns = {}
        for vn, vl_ in [("chart","Chart"),("surface","Vol Srf"),("portfolio","Portfolio"),("greeks","Options")]:
            b = QPushButton(vl_); b.setCheckable(True); b.setChecked(vn == "chart"); self.view_btns[vn] = b; vlb.addWidget(b)
        cl.addWidget(vg)

        # Tabs
        tabs = QTabWidget(); tabs.setStyleSheet(STYLESHEET)

        # Market tab
        mt = QWidget(); mtl = QVBoxLayout(mt); mtl.setSpacing(6); mtl.setContentsMargins(8, 10, 8, 8)
        mtl.addWidget(self._make_label("Ticker"))
        self.ticker_combo = QComboBox()
        for t, info in TICKERS.items(): self.ticker_combo.addItem(f"{t} — {info['name']}")
        mtl.addWidget(self.ticker_combo)
        mtl.addWidget(self._make_label("Period"))
        self.period_combo = QComboBox(); self.period_combo.addItems(["1 Month","3 Months","6 Months","1 Year","2 Years"]); self.period_combo.setCurrentIndex(3)
        mtl.addWidget(self.period_combo)
        mtl.addWidget(self._make_label("Overlays"))
        self.sma20_cb = QCheckBox("SMA 20"); self.sma50_cb = QCheckBox("SMA 50"); self.ema12_cb = QCheckBox("EMA 12"); self.bb_cb = QCheckBox("Bollinger (20,2)")
        for cb in [self.sma20_cb, self.sma50_cb, self.ema12_cb, self.bb_cb]: mtl.addWidget(cb)
        mtl.addWidget(self._make_label("Statistics"))
        self.stats_text = QTextEdit(); self.stats_text.setReadOnly(True); self.stats_text.setMinimumHeight(140); mtl.addWidget(self.stats_text); mtl.addStretch()
        tabs.addTab(mt, "Market")

        # Options tab
        ot = QWidget(); otl = QVBoxLayout(ot); otl.setSpacing(6); otl.setContentsMargins(8, 10, 8, 8)
        otl.addWidget(self._make_label("Type"))
        og = QWidget(); ogl = QHBoxLayout(og); ogl.setContentsMargins(0, 0, 0, 0); ogl.setSpacing(3)
        self.call_btn = QPushButton("CALL"); self.call_btn.setCheckable(True); self.call_btn.setChecked(True)
        self.put_btn = QPushButton("PUT"); self.put_btn.setCheckable(True); ogl.addWidget(self.call_btn); ogl.addWidget(self.put_btn); otl.addWidget(og)
        otl.addWidget(self._make_label("Strike"))
        self.strike_spin = QDoubleSpinBox(); self.strike_spin.setRange(1, 100000); self.strike_spin.setValue(185.0); self.strike_spin.setPrefix("$"); self.strike_spin.setDecimals(2); self.strike_spin.setSingleStep(5.0); otl.addWidget(self.strike_spin)
        otl.addWidget(self._make_label("DTE"))
        self.dte_spin = QSpinBox(); self.dte_spin.setRange(1, 730); self.dte_spin.setValue(30); self.dte_spin.setSuffix(" days"); otl.addWidget(self.dte_spin)
        otl.addWidget(self._make_label("IV"))
        self.iv_slider = QSlider(Qt.Horizontal); self.iv_slider.setRange(5, 150); self.iv_slider.setValue(25)
        self.iv_label = QLabel("25.0%"); self.iv_label.setStyleSheet(f"color:{ACCENT};font-size:12px;font-weight:bold;background:transparent;")
        otl.addWidget(self.iv_slider); otl.addWidget(self.iv_label)
        otl.addWidget(self._make_label("Rate"))
        self.rate_spin = QDoubleSpinBox(); self.rate_spin.setRange(0, 20); self.rate_spin.setValue(5.0); self.rate_spin.setSuffix("%"); self.rate_spin.setDecimals(2); self.rate_spin.setSingleStep(0.25); otl.addWidget(self.rate_spin)
        self.price_btn = QPushButton("⟐  PRICE OPTION"); self.price_btn.setStyleSheet(f"QPushButton{{background:{ACCENT_DIM};border-color:{ACCENT};color:{ACCENT};font-size:11px;padding:10px;}}"); otl.addWidget(self.price_btn)
        self.surface_btn = QPushButton("◇  GENERATE VOL SURFACE"); otl.addWidget(self.surface_btn)
        otl.addWidget(self._make_label("Result"))
        self.opt_result = QTextEdit(); self.opt_result.setReadOnly(True); self.opt_result.setMinimumHeight(100); otl.addWidget(self.opt_result); otl.addStretch()
        tabs.addTab(ot, "Options")

        # Portfolio tab
        pt = QWidget(); ptl = QVBoxLayout(pt); ptl.setSpacing(6); ptl.setContentsMargins(8, 10, 8, 8)
        ptl.addWidget(self._make_label("Portfolio")); self.load_sample = QPushButton("Load Sample Portfolio"); ptl.addWidget(self.load_sample)
        ptl.addWidget(self._make_label("Summary")); self.port_summary = QTextEdit(); self.port_summary.setReadOnly(True); self.port_summary.setMinimumHeight(100); ptl.addWidget(self.port_summary); ptl.addStretch()
        tabs.addTab(pt, "Portfolio")

        # Log tab
        it = QWidget(); itl = QVBoxLayout(it); itl.setSpacing(6); itl.setContentsMargins(8, 10, 8, 8)
        itl.addWidget(self._make_label("Log"))
        self.log_text = QTextEdit(); self.log_text.setReadOnly(True); self.log_text.setPlainText("[INFO] QuantView initialised\n[INFO] Offscreen FBO + QPainter\n[INFO] Ready\n")
        itl.addWidget(self.log_text); itl.addStretch(); tabs.addTab(it, "Log")

        cl.addWidget(tabs); cs.setWidget(ci)
        cpl = QVBoxLayout(cp); cpl.setContentsMargins(0, 0, 0, 0); cpl.addWidget(cs)

        self.viewport = FinanceViewport()
        self.main_layout.addWidget(cp); self.main_layout.addWidget(self.viewport, 1)

    # ── Signal Connections ────────────────────────────────────
    def _connect_signals(self):
        for vn, btn in self.view_btns.items():
            btn.clicked.connect(lambda c, v=vn: self.set_view(v))

        self.ticker_combo.currentTextChanged.connect(self.load_ticker)
        self.period_combo.currentTextChanged.connect(lambda: self.load_ticker(self.ticker_combo.currentText()))
        for cb in [self.sma20_cb, self.sma50_cb, self.ema12_cb, self.bb_cb]:
            cb.toggled.connect(lambda: self.update_overlays())

        self.call_btn.clicked.connect(self._on_call)
        self.put_btn.clicked.connect(self._on_put)
        self.iv_slider.valueChanged.connect(lambda v: self.iv_label.setText(f"{v:.1f}%"))

        self.price_btn.clicked.connect(self.on_price)
        self.surface_btn.clicked.connect(self.on_surface)
        self.load_sample.clicked.connect(lambda: self.load_portfolio(SAMPLE_PORTFOLIO))

    # ── Logic ─────────────────────────────────────────────────
    def set_view(self, v):
        self.viewport.view_mode = v
        for k, b in self.view_btns.items(): b.setChecked(k == v)
        self.viewport.update()

    def load_ticker(self, ct):
        tk = ct.split(" — ")[0].strip(); self.current_ticker = tk
        if tk not in TICKERS: return
        info = TICKERS[tk]; per = self.PERIOD_DAYS.get(self.period_combo.currentText(), 365)
        ck = f"{tk}_{per}"
        if ck not in self.all_td: self.all_td[ck] = generate_ohlcv(tk, days=per, base_price=info['base'], volatility=info['vol'], drift=info['drift'], seed=info['seed'])
        self.current_data = self.all_td[ck]; self.viewport.chart_data = self.current_data; self.viewport.ticker_label = tk; self.viewport.ticker_name = info['name']
        self.viewport.update(); self.update_stats(); self.update_overlays()
        if self.current_data: self.strike_spin.setValue(self.current_data[-1]['close'])
        self.log_text.append(f"[LOAD] {tk} ({len(self.current_data)} bars)")

    def update_stats(self):
        if not self.current_data or len(self.current_data) < 2: return
        c = [d['close'] for d in self.current_data]; l = c[-1]; f_ = c[0]; tr = (l - f_) / f_ * 100
        dr = [(c[i] - c[i-1]) / c[i-1] for i in range(1, len(c))]; ar = sum(dr) / len(dr)
        sr = math.sqrt(sum((r - ar)**2 for r in dr) / len(dr)); av = sr * math.sqrt(252) * 100
        sh = (ar * 252 - 0.05) / (sr * math.sqrt(252)) if sr > 0 else 0
        pk = c[0]; mx = 0
        for v in c:
            if v > pk: pk = v
            dd = (pk - v) / pk
            if dd > mx: mx = dd
        self.stats_text.setPlainText(f"Last:     ${l:.2f}\nReturn:   {'+' if tr>=0 else ''}{tr:.2f}%\nAnn Vol:  {av:.1f}%\nSharpe:   {sh:.3f}\nMax DD:   {mx*100:.1f}%\nBars:     {len(self.current_data)}\nSector:   {TICKERS.get(self.current_ticker,{}).get('sector','—')}")

    def update_overlays(self):
        if not self.current_data: return
        c = [d['close'] for d in self.current_data]; ov = {}
        if self.sma20_cb.isChecked(): ov['sma20'] = compute_sma(c, 20)
        if self.sma50_cb.isChecked(): ov['sma50'] = compute_sma(c, 50)
        if self.ema12_cb.isChecked(): ov['ema12'] = compute_ema(c, 12)
        if self.bb_cb.isChecked(): u, m, l = compute_bollinger(c, 20, 2); ov['bb_upper'] = u; ov['bb_mid'] = m; ov['bb_lower'] = l
        self.viewport.overlays = ov; self.viewport.update()

    def _on_call(self, c):
        if c: self.put_btn.setChecked(False)
        elif not self.put_btn.isChecked(): self.call_btn.setChecked(True)

    def _on_put(self, c):
        if c: self.call_btn.setChecked(False)
        elif not self.call_btn.isChecked(): self.put_btn.setChecked(True)

    def on_price(self):
        if not self.current_data: return
        S = self.current_data[-1]['close']; K = self.strike_spin.value(); T = self.dte_spin.value() / 365.0; r = self.rate_spin.value() / 100.0; sigma = self.iv_slider.value() / 100.0
        ot_ = 'call' if self.call_btn.isChecked() else 'put'; res = black_scholes(S, K, T, r, sigma, ot_)
        self.opt_result.setPlainText(f"{ot_.upper()} S=${S:.2f} K=${K:.2f}\nPrice: ${res['price']:.4f}\nΔ={res['delta']:.4f} Γ={res['gamma']:.6f}\nΘ={res['theta']:.4f} ν={res['vega']:.4f}")
        ladder = []
        for off in range(-6, 7):
            k = round(S + off * S * 0.025, 2)
            if k <= 0: continue
            cc = black_scholes(S, k, T, r, sigma, 'call'); pp = black_scholes(S, k, T, r, sigma, 'put')
            m = math.log(k / S); iv_a = sigma + 0.15 * m**2 - 0.05 * m
            ladder.append({'strike':k, 'call_delta':cc['delta'], 'call_price':cc['price'], 'iv':iv_a, 'put_price':pp['price'], 'put_delta':pp['delta']})
        self.viewport.greeks_data = {'main':{'price':res['price'],'delta':res['delta'],'gamma':res['gamma'],'theta':res['theta'],'vega':res['vega'],'rho':res['rho'],'type':ot_.upper(),'strike':K},'ladder':ladder,'spot':S}
        self.set_view('greeks'); self.log_text.append(f"[OPT] {ot_.upper()} K=${K:.2f} → ${res['price']:.4f}")

    def on_surface(self):
        if not self.current_data: return
        S = self.current_data[-1]['close']; self.viewport.vol_surface = generate_vol_surface(S, r=self.rate_spin.value() / 100.0, base_vol=self.iv_slider.value() / 100.0)
        self.viewport._ensure_gl(); self.viewport.upload_surface()
        self.set_view('surface'); self.log_text.append("[SURF] Vol surface generated")

    def load_portfolio(self, pf):
        for pos in pf:
            t = pos['ticker']; ck = f"{t}_365"
            if ck not in self.all_td and t in TICKERS:
                info = TICKERS[t]; self.all_td[ck] = generate_ohlcv(t, days=365, base_price=info['base'], volatility=info['vol'], drift=info['drift'], seed=info['seed'])
        td = {pos['ticker']:self.all_td.get(f"{pos['ticker']}_365", []) for pos in pf}
        st = compute_portfolio_stats(pf, td); self.viewport.portfolio_stats = st; self.set_view('portfolio')
        self.port_summary.setPlainText(f"Total: ${st['total_value']:,.2f}\nP&L: {'+'if st['total_pnl']>=0 else ''}${st['total_pnl']:,.2f}")
        self.log_text.append(f"[PORT] {len(pf)} positions, ${st['total_value']:,.2f}")


# ═══════════════════════════════════════════════════════════════
#  INSTANTIATE
# ═══════════════════════════════════════════════════════════════

quant_app = QuantViewApp()

# ═══════════════════════════════════════════════════════════════
#  ADD TO SCENE
# ═══════════════════════════════════════════════════════════════

quant_proxy = graphics_scene.addWidget(quant_app.main_widget)
quant_app.main_widget.resize(1400, 850)
view_rect = graphics_view.mapToScene(graphics_view.viewport().rect()).boundingRect()
quant_proxy.setPos(view_rect.center().x() - 700, view_rect.center().y() - 425)