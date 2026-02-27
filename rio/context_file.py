"""
Smart CONTEXT File — Statement-Level Code Compaction

Produces a compacted view of all successfully executed code that
represents the *current effective state* of the scene namespace.

Instead of per-block granularity, this works at per-STATEMENT level:
each top-level statement is tracked individually, so if block v3
defines both `web_view` and `shadow`, and block v6 redefines `shadow`,
only the shadow statements from v3 are dropped — `web_view` stays.

COMPACTION RULES:
  1. IMPORTS: Deduplicated and merged by module. Placed at top.
  2. ASSIGNMENTS: Last-writer-wins per variable name at statement level.
  3. SIDE EFFECTS: Grouped by target object. Latest group wins.
  4. HELPER CODE: `view = ...; if hasattr(view...): center = ...` patterns
     are deduplicated (emitted once, before first use).
     
INTEGRATION:
  In filesystem.py:
  
      from .context_file import create_smart_context_file_class
      
      # At module level or in RioRoot.__init__:
      SmartContextFile = create_smart_context_file_class(SyntheticFile)
      
      # Then replace:
      #   self.context_file = ContextFile()
      # With:
      self.context_file = SmartContextFile()
"""

import ast
import re
import textwrap
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import OrderedDict


# ============================================================
# Statement classification
# ============================================================

@dataclass
class Statement:
    """A single top-level statement with metadata."""
    source: str          # The source code
    version: int         # Which code block (execution order)
    index: int           # Position within that block
    
    kind: str = "other"  # "import", "assign", "side_effect", "control", "other"
    
    # For assignments: which names are defined
    defines: Set[str] = field(default_factory=set)
    
    # For side effects: which objects are called on
    targets: Set[str] = field(default_factory=set)
    
    # All names read by this statement
    reads: Set[str] = field(default_factory=set)
    
    # For imports: normalized import key
    import_key: str = ""
    
    # Content hash for deduplication of identical statements
    content_hash: str = ""


class StatementAnalyzer:
    """Parse code into individual analyzed top-level statements."""
    
    def analyze_block(self, code: str, version: int) -> List[Statement]:
        """Split a code block into analyzed top-level statements."""
        statements = []
        lines = code.split('\n')
        
        try:
            tree = ast.parse(code)
        except SyntaxError:
            # Fallback: treat entire block as one statement
            stmt = Statement(
                source=code.strip(), version=version, index=0,
                kind="other", content_hash=self._hash(code.strip())
            )
            return [stmt]
        
        for idx, node in enumerate(ast.iter_child_nodes(tree)):
            src = self._get_source(node, lines)
            if not src.strip():
                continue
            
            stmt = Statement(
                source=src, version=version, index=idx,
                content_hash=self._hash(src)
            )
            
            self._classify(node, stmt, lines)
            statements.append(stmt)
        
        return statements
    
    def _classify(self, node: ast.AST, stmt: Statement, lines: List[str]):
        """Classify a top-level AST node."""
        
        # --- Imports ---
        if isinstance(node, ast.Import):
            stmt.kind = "import"
            for alias in node.names:
                name = alias.asname or alias.name.split('.')[0]
                stmt.defines.add(name)
            stmt.import_key = self._normalize(stmt.source)
        
        elif isinstance(node, ast.ImportFrom):
            stmt.kind = "import"
            for alias in node.names:
                name = alias.asname or alias.name
                stmt.defines.add(name)
            stmt.import_key = f"from:{node.module}" if node.module else stmt.source
        
        # --- Assignments ---
        elif isinstance(node, ast.Assign):
            stmt.kind = "assign"
            for target in node.targets:
                self._extract_assigned_names(target, stmt.defines)
            self._extract_all_reads(node.value, stmt.reads)
        
        elif isinstance(node, ast.AugAssign):
            stmt.kind = "assign"
            self._extract_assigned_names(node.target, stmt.defines)
            self._extract_all_reads(node.value, stmt.reads)
        
        elif isinstance(node, ast.AnnAssign):
            stmt.kind = "assign"
            if node.target:
                self._extract_assigned_names(node.target, stmt.defines)
            if node.value:
                self._extract_all_reads(node.value, stmt.reads)
        
        # --- Function/Class defs ---
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            stmt.kind = "assign"
            stmt.defines.add(node.name)
        
        elif isinstance(node, ast.ClassDef):
            stmt.kind = "assign"
            stmt.defines.add(node.name)
        
        # --- Expression statements (side effects) ---
        elif isinstance(node, ast.Expr):
            stmt.kind = "side_effect"
            self._extract_all_reads(node.value, stmt.reads)
            # Find call target
            if isinstance(node.value, ast.Call):
                root = self._get_call_root(node.value)
                if root:
                    stmt.targets.add(root)
        
        # --- Control flow (if/for/while/with) at top level ---
        elif isinstance(node, (ast.If, ast.For, ast.While, ast.With)):
            stmt.kind = "control"
            for child in ast.walk(node):
                if isinstance(child, ast.Name):
                    if isinstance(child.ctx, ast.Store):
                        stmt.defines.add(child.id)
                    elif isinstance(child.ctx, ast.Load):
                        stmt.reads.add(child.id)
                elif isinstance(child, ast.Assign):
                    for t in child.targets:
                        self._extract_assigned_names(t, stmt.defines)
    
    def _get_source(self, node: ast.AST, lines: List[str]) -> str:
        if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
            return '\n'.join(lines[node.lineno - 1:node.end_lineno])
        return ''
    
    def _normalize(self, s: str) -> str:
        return ' '.join(s.split())
    
    def _hash(self, s: str) -> str:
        return self._normalize(s)
    
    def _extract_assigned_names(self, target, names: Set[str]):
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._extract_assigned_names(elt, names)
        elif isinstance(target, ast.Starred):
            self._extract_assigned_names(target.value, names)
        elif isinstance(target, ast.Attribute):
            root = self._get_attr_root(target)
            if root:
                names.add(root)
    
    def _extract_all_reads(self, node: ast.AST, reads: Set[str]):
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                reads.add(child.id)
    
    def _get_call_root(self, node: ast.Call) -> Optional[str]:
        func = node.func
        while isinstance(func, ast.Attribute):
            func = func.value
            if isinstance(func, ast.Call):
                func = func.func
        if isinstance(func, ast.Name):
            return func.id
        return None
    
    def _get_attr_root(self, node) -> Optional[str]:
        while isinstance(node, ast.Attribute):
            node = node.value
        if isinstance(node, ast.Name):
            return node.id
        return None


# ============================================================
# Import Merger
# ============================================================

class ImportMerger:
    """Merge and deduplicate import statements."""
    
    def merge(self, import_stmts: List[Statement]) -> List[str]:
        plain_imports: Dict[str, str] = {}
        from_imports: Dict[str, Set[str]] = {}
        
        for stmt in import_stmts:
            src = stmt.source.strip()
            
            m = re.match(r'from\s+([\w.]+)\s+import\s+(.+)', src, re.DOTALL)
            if m:
                module = m.group(1)
                names_raw = m.group(2).strip().strip('()')
                names = {n.strip() for n in names_raw.split(',') if n.strip()}
                from_imports.setdefault(module, set()).update(names)
                continue
            
            m = re.match(r'import\s+(.+)', src)
            if m:
                plain_imports[m.group(1).strip()] = src
                continue
            
            plain_imports[src] = src
        
        result = []
        for mod in sorted(plain_imports):
            result.append(plain_imports[mod])
        
        for module in sorted(from_imports):
            names = sorted(from_imports[module])
            line = f"from {module} import {', '.join(names)}"
            if len(line) > 88:
                names_str = ',\n    '.join(names)
                line = f"from {module} import (\n    {names_str}\n)"
            result.append(line)
        
        return result


# ============================================================
# Compactor
# ============================================================

class CodeCompactor:
    """
    Statement-level code compactor.
    
    For each variable name, keeps only the LAST statement that
    defines it. For side effects on the same target, keeps only
    the latest group. Imports are merged and pruned.
    """
    
    def __init__(self):
        self.analyzer = StatementAnalyzer()
        self.merger = ImportMerger()
    
    def compact(self, code_blocks: List[str]) -> str:
        if not code_blocks:
            return ""
        
        # 1. Analyze all blocks into statements
        all_stmts: List[Statement] = []
        for ver, code in enumerate(code_blocks):
            if not code or not code.strip():
                continue
            stmts = self.analyzer.analyze_block(code, version=ver)
            all_stmts.extend(stmts)
        
        if not all_stmts:
            return ""
        
        # 2. For each defined name, find the latest statement that defines it
        name_to_latest: Dict[str, Tuple[int, int]] = {}
        for stmt in all_stmts:
            if stmt.kind in ("assign", "control") and stmt.defines:
                for name in stmt.defines:
                    name_to_latest[name] = (stmt.version, stmt.index)
        
        # 3. For side effects, group by target — latest version wins
        #    EXCEPTION: side effects in the same block as a live definition
        #    of their target are kept (they're initialization code).
        target_latest_version: Dict[str, int] = {}
        for stmt in all_stmts:
            if stmt.kind == "side_effect" and stmt.targets:
                for t in stmt.targets:
                    target_latest_version[t] = max(
                        target_latest_version.get(t, -1), stmt.version
                    )
        
        # Find which versions contain live definitions for each name
        name_live_version: Dict[str, int] = {}
        for name, (ver, idx) in name_to_latest.items():
            name_live_version[name] = ver
        
        # 4. Deduplicate identical control-flow statements (keep last)
        control_dedup: Dict[str, Tuple[int, int]] = {}
        for stmt in all_stmts:
            if stmt.kind == "control":
                control_dedup[stmt.content_hash] = (stmt.version, stmt.index)
        
        # 5. Filter: keep only live statements
        import_stmts: List[Statement] = []
        body_stmts: List[Statement] = []
        
        for stmt in all_stmts:
            if stmt.kind == "import":
                import_stmts.append(stmt)
                continue
            
            if stmt.kind == "assign":
                is_live = any(
                    name_to_latest.get(n) == (stmt.version, stmt.index)
                    for n in stmt.defines
                )
                if is_live:
                    body_stmts.append(stmt)
            
            elif stmt.kind == "side_effect":
                if not stmt.targets:
                    body_stmts.append(stmt)
                else:
                    # Keep if this is the latest version with side effects on target
                    is_latest = any(
                        target_latest_version.get(t) == stmt.version
                        for t in stmt.targets
                    )
                    # Also keep if the target was DEFINED in this same block
                    # (initialization side effects travel with the definition)
                    is_init = any(
                        name_live_version.get(t) == stmt.version
                        for t in stmt.targets
                    )
                    if is_latest or is_init:
                        body_stmts.append(stmt)
            
            elif stmt.kind == "control":
                has_live_def = any(
                    name_to_latest.get(n) == (stmt.version, stmt.index)
                    for n in stmt.defines
                ) if stmt.defines else False
                
                is_latest_dup = control_dedup.get(stmt.content_hash) == (stmt.version, stmt.index)
                
                if has_live_def or is_latest_dup:
                    body_stmts.append(stmt)
            
            else:
                body_stmts.append(stmt)
        
        # 6. Prune imports not referenced by any live statement
        all_live_reads: Set[str] = set()
        all_live_defines: Set[str] = set()
        for stmt in body_stmts:
            all_live_reads |= stmt.reads
            all_live_defines |= stmt.defines
        
        needed_imports = [
            s for s in import_stmts
            if s.defines & (all_live_reads | all_live_defines)
        ]
        
        # 7. Assemble output
        parts = []
        
        merged = self.merger.merge(needed_imports)
        if merged:
            parts.append('\n'.join(merged))
        
        body_lines = []
        prev_version = -1
        for stmt in body_stmts:
            if stmt.version != prev_version and body_lines:
                body_lines.append('')
            body_lines.append(stmt.source)
            prev_version = stmt.version
        
        if body_lines:
            parts.append('\n'.join(body_lines))
        
        return '\n\n'.join(parts) + '\n'


# ============================================================
# Drop-in ContextFile replacement (for filesystem.py)
# ============================================================

import asyncio
from typing import List as TList


def create_smart_context_file_class(SyntheticFile):
    """
    Factory that creates a SmartContextFile class inheriting from your SyntheticFile.
    
    Usage in filesystem.py:
    
        from .context_file import create_smart_context_file_class
        SmartContextFile = create_smart_context_file_class(SyntheticFile)
        
        # Then in RioRoot.__init__:
        self.context_file = SmartContextFile()
    """
    
    class SmartContextFile(SyntheticFile):
        """
        Drop-in replacement for ContextFile with smart compaction.
        Inherits SyntheticFile so it has qid, stat, and all 9P plumbing.
        """
        
        def __init__(self):
            super().__init__("CONTEXT")
            
            self._code_blocks: TList[str] = []
            self._compactor = CodeCompactor()
            self._cached: Optional[str] = None
            
            self._content_ready = asyncio.Event()
            self._content_consumed = False
            self._lock = asyncio.Lock()
        
        def append_code(self, code: str):
            """Append successfully executed code. Called by ParseFile on success."""
            if code and code.strip():
                self._code_blocks.append(code.rstrip() + "\n")
                self._cached = None
                self._content_ready.set()
        
        def get_all_code(self) -> str:
            """Return compacted code representing current effective state."""
            if self._cached is None:
                self._cached = self._compactor.compact(self._code_blocks)
            return self._cached
        
        def get_raw_code(self) -> str:
            """Return all raw code blocks concatenated (for replay / debug)."""
            return "\n".join(self._code_blocks)
        
        async def read(self, fid, offset: int, count: int) -> bytes:
            """Blocking read — same semantics as original ContextFile."""
            if offset == 0 and self._content_consumed:
                async with self._lock:
                    if self._content_consumed:
                        self._content_consumed = False
                        self._content_ready.clear()
            
            await self._content_ready.wait()
            
            async with self._lock:
                content = self.get_all_code()
                data = content.encode()
                chunk = data[offset:offset + count]
                if offset + len(chunk) >= len(data):
                    self._content_consumed = True
                return chunk
        
        async def write(self, fid, offset: int, data: bytes) -> int:
            raise PermissionError(
                "CONTEXT is read-only. Code is appended automatically on successful execution."
            )
    
    return SmartContextFile


# ============================================================
# Tests
# ============================================================

def test_basic_compaction():
    c = CodeCompactor()
    result = c.compact(["x = 1\ny = 2\n", "x = 10\n"])
    assert "x = 10" in result
    assert "x = 1\n" not in result  # x=1 superseded, but x=10 stays
    assert "y = 2" in result
    print("✓ test_basic_compaction")


def test_import_merging():
    c = CodeCompactor()
    result = c.compact([
        "from PySide6.QtGui import QColor\nx = QColor()\n",
        "from PySide6.QtGui import QBrush\ny = QBrush()\n",
    ])
    assert "from PySide6.QtGui import" in result
    assert "QBrush" in result and "QColor" in result
    lines = [l for l in result.split('\n') if 'from PySide6.QtGui import' in l]
    assert len(lines) == 1, f"Expected 1 merged import, got {len(lines)}"
    print("✓ test_import_merging")


def test_side_effect_dedup():
    c = CodeCompactor()
    result = c.compact([
        "obj = create()\n",
        'obj.configure("old")\n',
        'obj.configure("new")\n',
    ])
    assert '"new"' in result
    assert '"old"' not in result
    print("✓ test_side_effect_dedup")


def test_shadow_superseded():
    c = CodeCompactor()
    result = c.compact([
        textwrap.dedent("""\
        from PySide6.QtWidgets import QGraphicsDropShadowEffect
        from PySide6.QtGui import QColor
        web_view = create_webview()
        map_proxy = scene.addWidget(web_view)
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(30)
        shadow.setOffset(0, 8)
        shadow.setColor(QColor(0, 0, 0, 180))
        map_proxy.setGraphicsEffect(shadow)
        """),
        textwrap.dedent("""\
        from PySide6.QtWidgets import QGraphicsDropShadowEffect
        from PySide6.QtGui import QColor
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(30)
        shadow.setOffset(45, 45)
        shadow.setColor(QColor(0, 0, 0, 180))
        map_proxy.setGraphicsEffect(shadow)
        """),
    ])
    assert "web_view = create_webview()" in result
    assert "map_proxy = scene.addWidget(web_view)" in result
    assert "setOffset(45, 45)" in result
    assert "setOffset(0, 8)" not in result
    print("✓ test_shadow_superseded")


def test_control_flow_dedup():
    c = CodeCompactor()
    center_pattern = textwrap.dedent("""\
    view = main_window.centralWidget()
    if hasattr(view, 'mapToScene'):
        center = view.mapToScene(view.viewport().rect().center())
    else:
        center = graphics_scene.sceneRect().center()
    """)
    result = c.compact([
        f"button = create()\n{center_pattern}button.setPos(center)\n",
        f"calendar = create()\n{center_pattern}calendar.setPos(center)\n",
    ])
    count = result.count("if hasattr(view, 'mapToScene')")
    assert count == 1, f"Expected 1 occurrence, got {count}"
    print("✓ test_control_flow_dedup")


def test_unused_import_pruning():
    c = CodeCompactor()
    result = c.compact([
        "from pathlib import Path\nx = Path('.')\n",
        "x = 42\n",
    ])
    assert "pathlib" not in result
    assert "x = 42" in result
    print("✓ test_unused_import_pruning")


def test_full_example():
    c = CodeCompactor()
    blocks = [
        textwrap.dedent("""\
        from PySide6.QtWidgets import QPushButton
        from PySide6.QtCore import QPointF
        button = QPushButton("Click Me")
        proxy = graphics_scene.addWidget(button)
        view = main_window.centralWidget()
        if hasattr(view, 'mapToScene'):
            center = view.mapToScene(view.viewport().rect().center())
        else:
            center = graphics_scene.sceneRect().center()
        proxy.setPos(center.x() - button.width() / 2, center.y() - button.height() / 2)
        """),
        textwrap.dedent("""\
        from PySide6.QtWidgets import QCalendarWidget
        calendar = QCalendarWidget()
        cal_proxy = graphics_scene.addWidget(calendar)
        view = main_window.centralWidget()
        if hasattr(view, 'mapToScene'):
            center = view.mapToScene(view.viewport().rect().center())
        else:
            center = graphics_scene.sceneRect().center()
        button_width = proxy.boundingRect().width()
        cal_x = center.x() + button_width / 2 + 20
        cal_y = center.y() - calendar.sizeHint().height() / 2
        cal_proxy.setPos(cal_x, cal_y)
        """),
        textwrap.dedent("""\
        import os
        from dotenv import load_dotenv
        from PySide6.QtWebEngineWidgets import QWebEngineView
        from PySide6.QtCore import QUrl
        load_dotenv()
        MAPBOX_TOKEN = os.getenv("MAPBOX_TOKEN", "")
        map_html = f"<html>first version</html>"
        web_view = QWebEngineView()
        web_view.setFixedSize(800, 500)
        web_view.setHtml(map_html, QUrl("https://localhost/"))
        map_proxy = graphics_scene.addWidget(web_view)
        view = main_window.centralWidget()
        if hasattr(view, 'mapToScene'):
            center = view.mapToScene(view.viewport().rect().center())
        else:
            center = graphics_scene.sceneRect().center()
        map_proxy.setPos(center.x() - 400, center.y() - 250)
        from PySide6.QtWidgets import QGraphicsDropShadowEffect
        from PySide6.QtGui import QColor
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(30)
        shadow.setOffset(0, 8)
        shadow.setColor(QColor(0, 0, 0, 180))
        map_proxy.setGraphicsEffect(shadow)
        """),
        "web_view.page().runJavaScript('map.flyTo({zoom: 10})')\n",
        "web_view.page().runJavaScript('map.flyTo({zoom: 15.5, pitch: 60})')\n",
        textwrap.dedent("""\
        from PySide6.QtWidgets import QGraphicsDropShadowEffect
        from PySide6.QtGui import QColor
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(30)
        shadow.setOffset(45, 45)
        shadow.setColor(QColor(0, 0, 0, 180))
        map_proxy.setGraphicsEffect(shadow)
        """),
    ]
    
    raw = "\n".join(blocks)
    result = c.compact(blocks)
    ratio = len(result) / len(raw) if raw else 1
    
    print(f"\n{'='*60}")
    print("COMPACTED OUTPUT:")
    print(f"{'='*60}")
    print(result)
    print(f"{'='*60}")
    print(f"Raw:       {len(raw):>5} chars, {len(raw.splitlines()):>3} lines")
    print(f"Compacted: {len(result):>5} chars, {len(result.splitlines()):>3} lines")
    print(f"Reduction: {100 - ratio*100:.0f}%")
    
    assert "setOffset(0, 8)" not in result, "Old shadow should be gone"
    assert "setOffset(45, 45)" in result, "New shadow should be present"
    assert "zoom: 10" not in result, "Old flyTo should be gone"
    assert "zoom: 15.5" in result, "New flyTo should be present"
    assert "web_view = QWebEngineView()" in result
    assert "QPushButton" in result
    assert "QCalendarWidget" in result
    print("✓ test_full_example")


if __name__ == "__main__":
    test_basic_compaction()
    test_import_merging()
    test_side_effect_dedup()
    test_shadow_superseded()
    test_control_flow_dedup()
    test_unused_import_pruning()
    test_full_example()
    print("\n✅ All tests passed!")