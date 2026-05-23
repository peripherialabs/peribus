# rio.signals — useful patterns

Examples you can paste straight into a rio scene. Every snippet assumes
`subscribe("<peer>")` has already been called (or comes earlier in the
same code block) and that you have `graphics_scene` available, which
rio provides automatically.

**Two signal names live side-by-side in your namespace:**
- `Signal(...)` — PySide6's regular Signal. Local only. Same semantics
  as anywhere else you use PySide6.
- `RemoteSignal(...)` — rio's cross-machine version. Same API
  (`.connect` / `.emit` / `.disconnect`), but emits also reach every
  peer that has subscribed to you. This is what the examples below use.

All of these only need the **signal name** to match across machines.
What's emitted on machine A's `foo = RemoteSignal(...)` is delivered to
machine B's `foo = RemoteSignal(...)` (with the same args). Local Qt slots
also fire normally — every RemoteSignal wraps a real `PySide6.Signal`
under the hood.

---

## 1. Remote button → remote action

The hello-world. Machine A has a button; clicking it triggers something
on machine B. Same shape works for "trigger build", "open the door",
"play the next track" — anything where the click and the effect are
on different boxes.

### sender (machine A)

```python
from PySide6.QtWidgets import QPushButton

subscribe("B")

try:
    fire = RemoteSignal()                      # no payload
except NameError:
    pass                                  # re-running the cell

btn = QPushButton("Fire the cannon")
btn.clicked.connect(fire.emit)
graphics_scene.addWidget(btn).setPos(0, 0)
```

### receiver (machine B)

```python
subscribe("A")

fire = RemoteSignal()
def handle():
    print("BOOM (received from A)")
fire.connect(handle)
```

---

## 2. Shared pointer (mouse position broadcast)

Show every connected machine's mouse position as a colored dot on the
scene. This is the "shared whiteboard cursor" pattern — pair it with
a drawing tool and you have collaborative whiteboarding.

### everyone (run this on every peer)

```python
from PySide6.QtCore import QPointF, Qt, QObject, QEvent
from PySide6.QtWidgets import QGraphicsEllipseItem
from PySide6.QtGui import QBrush, QColor, QPen
import socket as _s

# Each peer broadcasts its cursor and renders incoming cursors.
me = _s.gethostname().split(".")[0]      # or hardcode your machine name
subscribe("ekanza")                       # subscribe to every peer you want
subscribe("cirno")                        # to see (skip self — it's a no-op)

# Outbound: send (who, x, y) every time the mouse moves on our view.
cursor_at = RemoteSignal(str, float, float)

class _PointerWatcher(QObject):
    """Installs an event filter on the graphics view; emits cursor_at."""
    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.MouseMove:
            scene_pt = graphics_view.mapToScene(ev.pos())
            cursor_at.emit(me, scene_pt.x(), scene_pt.y())
        return False

if "_watcher" not in dir():
    _watcher = _PointerWatcher()
    graphics_view.viewport().installEventFilter(_watcher)

# Inbound: one persistent dot per peer.
if "_peer_dots" not in dir():
    _peer_dots = {}    # who -> QGraphicsEllipseItem

def _on_cursor(who, x, y):
    if who == me:
        return                            # don't render our own
    dot = _peer_dots.get(who)
    if dot is None:
        # Stable color per peer name so it's consistent across reloads.
        h = abs(hash(who)) % 360
        col = QColor.fromHsv(h, 200, 240)
        dot = QGraphicsEllipseItem(-8, -8, 16, 16)
        dot.setBrush(QBrush(col))
        dot.setPen(QPen(Qt.black, 1))
        dot.setZValue(1000)
        graphics_scene.addItem(dot)
        _peer_dots[who] = dot
    dot.setPos(x, y)

cursor_at.connect(_on_cursor)
```

You'll see every peer's pointer slide around your scene in real time.
Each peer chooses what they call themselves via `me`; the wire only
carries names, so there's no central registry.

---

## 3. Shared clipboard

Anything copied on one machine becomes available on the others. The
"sync" is consensual — you opt in by subscribing.

### everyone

```python
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

subscribe("ekanza")
subscribe("cirno")

clipboard_text = RemoteSignal(str)

# Outbound: poll local clipboard every 500 ms. (Qt's clipboard signals
# don't always fire reliably across desktops; polling is simpler and
# the cost is negligible.) We remember what we last sent so we don't
# spam the wire with the same string.
_last_sent = {"v": None}
def _poll_clipboard():
    cb = QApplication.clipboard()
    cur = cb.text()
    if cur and cur != _last_sent["v"]:
        _last_sent["v"] = cur
        clipboard_text.emit(cur)

if "_clip_timer" not in dir():
    _clip_timer = QTimer()
    _clip_timer.timeout.connect(_poll_clipboard)
    _clip_timer.start(500)

# Inbound: write into local clipboard. Suppress the echo loop by
# stashing what came in as "_last_sent" so the next poll ignores it.
def _on_clipboard(text):
    if text == _last_sent["v"]:
        return
    _last_sent["v"] = text
    QApplication.clipboard().setText(text)

clipboard_text.connect(_on_clipboard)
```

Copy "hello world" on ekanza, paste it on cirno. The Qt clipboard
carries plain text here; for HTML / images you'd emit a `bytes` payload
and use `setMimeData` instead — same shape, bigger encoding.

---

## 4. Distributed counter / dashboard

Every machine maintains a local count of something (open issues, build
errors, current temperature). One "dashboard" machine aggregates and
shows the total. Useful for status boards.

### producers (each non-dashboard peer)

```python
import os, time
from PySide6.QtCore import QTimer
import socket as _s

subscribe("dashboard")

me = _s.gethostname().split(".")[0]
metric_report = RemoteSignal(str, str, float)   # (machine, metric, value)

def _push():
    # whatever you actually want to report
    load = os.getloadavg()[0]
    metric_report.emit(me, "load1", load)

if "_metric_timer" not in dir():
    _metric_timer = QTimer()
    _metric_timer.timeout.connect(_push)
    _metric_timer.start(2000)
```

### dashboard

```python
from PySide6.QtWidgets import QLabel
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

subscribe("ekanza")
subscribe("cirno")
# ... subscribe to every producer

metric_report = RemoteSignal(str, str, float)

if "_dash_labels" not in dir():
    _dash_labels = {}                     # (machine, metric) -> QLabel
    _dash_y = {"next": 30}

def _on_metric(machine, metric, value):
    key = (machine, metric)
    lbl = _dash_labels.get(key)
    if lbl is None:
        lbl = QLabel()
        lbl.setFont(QFont("monospace", 16))
        lbl.setStyleSheet("color: white; background: rgba(0,0,0,180); padding: 6px;")
        proxy = graphics_scene.addWidget(lbl)
        proxy.setPos(20, _dash_y["next"])
        _dash_y["next"] += 40
        _dash_labels[key] = lbl
    lbl.setText(f"{machine:>10s}  {metric:>8s}  {value:>8.2f}")

metric_report.connect(_on_metric)
```

The dashboard layout self-organizes as new machines report in.
Producers don't know who's watching; the dashboard doesn't need to
know which machines exist ahead of time.

---

## 5. Multiplayer "car" — synchronized scene actor

A square that anyone can drive with WASD. Their version moves, the
others see it move. Demonstrates the latency profile clearly — UDP +
JSON + Qt = sub-frame on a LAN.

### everyone

```python
from PySide6.QtWidgets import QGraphicsRectItem
from PySide6.QtCore import Qt, QEvent, QObject, QTimer
from PySide6.QtGui import QColor, QBrush, QPen
import socket as _s

me = _s.gethostname().split(".")[0]
subscribe("ekanza")
subscribe("cirno")

# Wire-level: each machine emits its own position whenever it changes.
car_state = RemoteSignal(str, float, float, float)   # (driver, x, y, heading)

# ── Local car ─────────────────────────────────────────────────────
if "_local_car" not in dir():
    _local_car = QGraphicsRectItem(-20, -10, 40, 20)
    h = abs(hash(me)) % 360
    _local_car.setBrush(QBrush(QColor.fromHsv(h, 200, 240)))
    _local_car.setPen(QPen(Qt.black, 1))
    _local_car.setPos(300, 300)
    _local_car._heading = 0.0
    graphics_scene.addItem(_local_car)

# Keyboard control. Install a viewport event filter so WASD works
# regardless of focus.
_keys = {"W": False, "A": False, "S": False, "D": False}
class _Driver(QObject):
    def eventFilter(self, obj, ev):
        if ev.type() in (QEvent.KeyPress, QEvent.KeyRelease):
            pressed = ev.type() == QEvent.KeyPress
            txt = ev.text().upper()
            if txt in _keys:
                _keys[txt] = pressed
        return False

if "_driver" not in dir():
    _driver = _Driver()
    graphics_view.viewport().installEventFilter(_driver)
    graphics_view.viewport().setFocusPolicy(Qt.StrongFocus)

# Physics tick — runs at 60 Hz. Cheap: just integrates velocity.
import math
def _tick():
    speed = 0.0
    turn = 0.0
    if _keys["W"]: speed += 4
    if _keys["S"]: speed -= 3
    if _keys["A"]: turn  -= 0.06
    if _keys["D"]: turn  += 0.06

    if speed == 0 and turn == 0:
        return  # don't broadcast if we're not moving

    _local_car._heading += turn
    dx = speed * math.cos(_local_car._heading)
    dy = speed * math.sin(_local_car._heading)
    _local_car.moveBy(dx, dy)
    _local_car.setRotation(math.degrees(_local_car._heading))

    car_state.emit(me, _local_car.x(), _local_car.y(), _local_car._heading)

if "_tick_timer" not in dir():
    _tick_timer = QTimer()
    _tick_timer.timeout.connect(_tick)
    _tick_timer.start(16)                # 60 Hz

# ── Remote cars ───────────────────────────────────────────────────
if "_remote_cars" not in dir():
    _remote_cars = {}     # driver_name -> QGraphicsRectItem

def _on_car(driver, x, y, heading):
    if driver == me:
        return
    car = _remote_cars.get(driver)
    if car is None:
        car = QGraphicsRectItem(-20, -10, 40, 20)
        h = abs(hash(driver)) % 360
        car.setBrush(QBrush(QColor.fromHsv(h, 200, 240)))
        car.setPen(QPen(Qt.black, 1))
        graphics_scene.addItem(car)
        _remote_cars[driver] = car
    car.setPos(x, y)
    car.setRotation(math.degrees(heading))

car_state.connect(_on_car)
```

WASD on one machine; cars from all peers drive around on every screen.
At 60 Hz you're sending ~960 bytes/s per car — irrelevant. If you ever
need to push that harder, sample at 30 Hz and interpolate on the
receiving side; the bus's wire is the same.

---

## 6. Remote function call (request/response)

Strict pub/sub is great for fire-and-forget, but sometimes you want a
reply. Easy with two signals:

### server (machine B)

```python
subscribe("A")

rpc_request  = RemoteSignal(str, str)   # (req_id, query)
rpc_response = RemoteSignal(str, str)   # (req_id, result)

def _handle(req_id, query):
    # do the work; this runs on B
    result = f"echo: {query.upper()}"
    rpc_response.emit(req_id, result)

rpc_request.connect(_handle)
```

### caller (machine A)

```python
import uuid

subscribe("B")

rpc_request  = RemoteSignal(str, str)
rpc_response = RemoteSignal(str, str)

_pending = {}                     # req_id -> callback

def call_remote(query, on_done):
    rid = uuid.uuid4().hex
    _pending[rid] = on_done
    rpc_request.emit(rid, query)

def _on_response(rid, result):
    cb = _pending.pop(rid, None)
    if cb: cb(result)

rpc_response.connect(_on_response)

# usage:
call_remote("hello", lambda r: print("got:", r))
```

`req_id` correlates request and response. Add a timeout if you need
to recover from a dead peer (just `QTimer.singleShot` an entry's
`_pending.pop`).

---

## 7. Distributed kill switch

One signal everyone subscribes to; anyone can fire it; everyone reacts.
Useful for "stop the demo", "save and quit", emergency clear.

```python
subscribe("ekanza"); subscribe("cirno"); # ... all peers

emergency = RemoteSignal(str)           # reason

def _on_emergency(reason):
    print(f"!! EMERGENCY: {reason}")
    # do whatever's appropriate — close popups, pause timers, etc.

emergency.connect(_on_emergency)

# fire from any peer:
# emergency.emit("disk filling up")
```

---

## Patterns to internalize

- **Pure-name pub/sub.** Both ends agree on a signal name and a type
  signature; nothing else is shared between processes. No imports,
  no classes, no proxies.
- **Subscribe is asymmetric.** `subscribe("X")` says "I want to hear
  X's emits". X doesn't need to subscribe back unless X also wants to
  hear yours.
- **Local emit still works.** Every cross-machine signal is also a
  normal local Qt signal. Connecting a UI slot AND a remote effect
  to the same signal is fine — they both fire.
- **The wire is fire-and-forget UDP.** Drop tolerance: cursor jitter
  during a packet loss is fine; financial transactions are not. For
  reliability over the same channel, send sequence numbers and
  resend on gap detection (or just use a TCP-backed file in rio).
- **Names live in your namespace.** If you want a "type" of signal
  scoped to a class, just declare it inside the class: each instance
  inherits the same class-attribute signal. Auto-naming uses the
  attribute name, so `class X: foo = RemoteSignal(int)` always exports
  as `foo`.

---

## Limits worth knowing

- Each emit serializes to JSON, must be under ~60 KB; for images,
  meshes, audio buffers — keep using a 9P file.
- No auth at the signal layer. Anyone on the network who knows your
  machine and signal port can subscribe and start receiving. The bus
  binds on `0.0.0.0` by default — bind it to your VPN interface in
  hostile networks.
- Type tuples are advisory at the receiver. If A emits `RemoteSignal(int)`
  and B declared `RemoteSignal(str)`, B will log a TypeError and discard
  the packet. The system fails loud rather than silently coercing.