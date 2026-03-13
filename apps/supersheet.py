"""
╔══════════════════════════════════════════════════════════════════╗
║  GRIDION — Advanced Spreadsheet Engine for Rio                  ║
║  A next-generation spreadsheet that runs inside the Rio parser  ║
║  with PySide6. Features: formula engine, cell styling,          ║
║  conditional formatting, charts, multi-sheet tabs, CSV/JSON     ║
║  import-export, undo/redo, find & replace, freeze panes,        ║
║  auto-fill, drag-select, and a polished light-mode UI.          ║
╚══════════════════════════════════════════════════════════════════╝
"""

import re
import math
import json
import csv
import io
import copy
import os
from datetime import datetime, timedelta
from collections import defaultdict
from functools import lru_cache

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QTabWidget, QTabBar, QPushButton, QLabel, QLineEdit,
    QComboBox, QToolBar, QToolButton, QMenu, QMenuBar,
    QStatusBar, QFrame, QDialog, QDialogButtonBox,
    QFileDialog, QMessageBox, QColorDialog, QFontDialog,
    QSpinBox, QDoubleSpinBox, QCheckBox, QTextEdit,
    QSplitter, QTreeWidget, QTreeWidgetItem, QScrollArea,
    QApplication, QStyledItemDelegate, QStyleOptionViewItem,
    QSizePolicy, QGraphicsDropShadowEffect, QInputDialog,
    QWidgetAction, QProgressBar
)
from PySide6.QtCore import (
    Qt, QTimer, QSize, QRect, QPoint, Signal, Slot,
    QPropertyAnimation, QEasingCurve, QParallelAnimationGroup,
    QSequentialAnimationGroup, QAbstractAnimation, Property
)
from PySide6.QtGui import (
    QColor, QBrush, QPen, QFont, QFontMetrics, QIcon,
    QPainter, QLinearGradient, QRadialGradient,
    QKeySequence, QAction, QShortcut, QPalette, QPixmap,
    QCursor, QTextCharFormat
)


# ─────────────────────────────────────────────
# COLOR PALETTE — Clean Light Theme
# ─────────────────────────────────────────────
class Theme:
    BG_DEEP       = "#F0F1F4"
    BG_PRIMARY    = "#FFFFFF"
    BG_SECONDARY  = "#F7F8FA"
    BG_TERTIARY   = "#ECEDF0"
    BG_CELL       = "#FFFFFF"
    BG_CELL_ALT   = "#FAFBFC"
    BG_SELECTED   = "#D6E4FF"
    BG_HOVER      = "#E8ECF2"
    BG_INPUT      = "#FFFFFF"

    ACCENT        = "#2563EB"     # blue-600
    ACCENT_DIM    = "#1D4ED8"     # blue-700
    ACCENT_GLOW   = "#3B82F6"     # blue-500
    SUCCESS       = "#16A34A"     # green-600
    WARNING       = "#D97706"     # amber-600
    DANGER        = "#DC2626"     # red-600
    PURPLE        = "#7C3AED"     # violet-600
    ORANGE        = "#EA580C"     # orange-600
    PINK          = "#DB2777"     # pink-600
    TEAL          = "#0D9488"     # teal-600

    TEXT_PRIMARY   = "#1E293B"    # slate-800
    TEXT_SECONDARY = "#475569"    # slate-600
    TEXT_DIM       = "#94A3B8"    # slate-400
    TEXT_ACCENT    = "#2563EB"

    BORDER         = "#D1D5DB"    # gray-300
    BORDER_FOCUS   = "#2563EB"
    BORDER_SUBTLE  = "#E5E7EB"    # gray-200

    HEADER_BG      = "#F1F3F5"
    HEADER_TEXT     = "#475569"   # slate-600

    TAB_ACTIVE     = "#2563EB"
    TAB_INACTIVE   = "#D1D5DB"

    SCROLLBAR_BG   = "#F0F1F4"
    SCROLLBAR_HANDLE = "#C4C9D2"

    CHART_COLORS = ["#2563EB", "#16A34A", "#D97706", "#DC2626",
                    "#7C3AED", "#EA580C", "#DB2777", "#0D9488"]


# ─────────────────────────────────────────────
# FORMULA ENGINE
# ─────────────────────────────────────────────
class FormulaEngine:
    """
    Spreadsheet formula engine supporting:
    - Cell references (A1, B2, $A$1, A$1, $A1)
    - Range references (A1:B5)
    - Functions: SUM, AVERAGE, MIN, MAX, COUNT, COUNTA, IF, AND, OR, NOT,
      CONCATENATE, LEN, LEFT, RIGHT, MID, UPPER, LOWER, TRIM, ROUND,
      ABS, POWER, SQRT, MOD, INT, VLOOKUP, HLOOKUP, INDEX, MATCH,
      SUMIF, COUNTIF, AVERAGEIF, NOW, TODAY, DATE, YEAR, MONTH, DAY,
      ISNUMBER, ISTEXT, ISBLANK, IFERROR, TEXT, VALUE, PI, RAND
    - Nested formulas
    - Circular reference detection
    """

    CELL_RE = re.compile(r'\$?([A-Z]+)\$?(\d+)', re.IGNORECASE)
    RANGE_RE = re.compile(r'\$?([A-Z]+)\$?(\d+):\$?([A-Z]+)\$?(\d+)', re.IGNORECASE)

    def __init__(self, get_cell_value_fn):
        self.get_cell = get_cell_value_fn
        self._evaluating = set()  # circular ref detection

    @staticmethod
    def col_to_index(col_str):
        """Convert column string to 0-based index: A->0, B->1, ..., Z->25, AA->26"""
        col_str = col_str.upper()
        result = 0
        for ch in col_str:
            result = result * 26 + (ord(ch) - ord('A') + 1)
        return result - 1

    @staticmethod
    def index_to_col(idx):
        """Convert 0-based index to column string"""
        result = ""
        idx += 1
        while idx > 0:
            idx -= 1
            result = chr(idx % 26 + ord('A')) + result
            idx //= 26
        return result

    def get_range_values(self, range_str):
        """Get all values from a range like A1:B5"""
        m = self.RANGE_RE.match(range_str)
        if not m:
            return []
        c1, r1, c2, r2 = m.groups()
        col1, col2 = self.col_to_index(c1), self.col_to_index(c2)
        row1, row2 = int(r1) - 1, int(r2) - 1
        if col1 > col2: col1, col2 = col2, col1
        if row1 > row2: row1, row2 = row2, row1
        values = []
        for r in range(row1, row2 + 1):
            for c in range(col1, col2 + 1):
                val = self.get_cell(r, c)
                values.append(val)
        return values

    def _parse_numeric(self, values):
        nums = []
        for v in values:
            if v is None or v == "":
                continue
            try:
                nums.append(float(v))
            except (ValueError, TypeError):
                continue
        return nums

    def evaluate(self, formula, row, col):
        """Evaluate a formula string starting with '='"""
        cell_key = (row, col)
        if cell_key in self._evaluating:
            return "#CIRC!"
        self._evaluating.add(cell_key)
        try:
            expr = formula[1:].strip()  # strip '='
            result = self._eval_expr(expr, row, col)
            return result
        except ZeroDivisionError:
            return "#DIV/0!"
        except Exception as e:
            return f"#ERR!"
        finally:
            self._evaluating.discard(cell_key)

    def _eval_expr(self, expr, row, col):
        """Evaluate an expression — replaces functions and refs then uses eval"""
        # Replace function calls
        processed = self._process_functions(expr, row, col)
        # Replace remaining cell references
        processed = self._replace_cell_refs(processed, row, col)
        # Safe eval
        try:
            result = eval(processed, {"__builtins__": {}}, {
                "math": math, "abs": abs, "round": round, "int": int,
                "float": float, "str": str, "len": len, "min": min,
                "max": max, "sum": sum, "True": True, "False": False,
                "pi": math.pi, "rand": __import__('random').random
            })
            return result
        except Exception:
            return "#ERR!"

    def _process_functions(self, expr, row, col):
        """Process spreadsheet functions recursively"""
        func_re = re.compile(r'([A-Z_]+)\s*\(', re.IGNORECASE)
        max_iter = 50
        iteration = 0
        while iteration < max_iter:
            iteration += 1
            m = func_re.search(expr)
            if not m:
                break
            func_name = m.group(1).upper()
            start = m.start()
            paren_start = m.end() - 1
            # Find matching close paren
            depth = 1
            i = paren_start + 1
            while i < len(expr) and depth > 0:
                if expr[i] == '(':
                    depth += 1
                elif expr[i] == ')':
                    depth -= 1
                i += 1
            if depth != 0:
                break
            paren_end = i - 1
            args_str = expr[paren_start + 1:paren_end]
            # Evaluate the function
            result = self._eval_function(func_name, args_str, row, col)
            expr = expr[:start] + str(result) + expr[paren_end + 1:]
        return expr

    def _split_args(self, args_str):
        """Split function arguments respecting nested parens and quotes"""
        args = []
        depth = 0
        current = ""
        in_str = False
        str_char = None
        for ch in args_str:
            if in_str:
                current += ch
                if ch == str_char:
                    in_str = False
                continue
            if ch in ('"', "'"):
                in_str = True
                str_char = ch
                current += ch
            elif ch == '(':
                depth += 1
                current += ch
            elif ch == ')':
                depth -= 1
                current += ch
            elif ch == ',' and depth == 0:
                args.append(current.strip())
                current = ""
            else:
                current += ch
        if current.strip():
            args.append(current.strip())
        return args

    def _eval_function(self, name, args_str, row, col):
        """Evaluate a spreadsheet function"""
        # Check for range arguments
        range_match = self.RANGE_RE.search(args_str)

        if name == "SUM":
            if range_match:
                vals = self.get_range_values(args_str.strip())
                return sum(self._parse_numeric(vals))
            else:
                args = self._split_args(args_str)
                total = 0
                for a in args:
                    v = self._eval_expr(a, row, col)
                    try: total += float(v)
                    except: pass
                return total

        elif name == "AVERAGE":
            if range_match:
                vals = self._parse_numeric(self.get_range_values(args_str.strip()))
                return sum(vals) / len(vals) if vals else "#DIV/0!"
            else:
                args = self._split_args(args_str)
                nums = []
                for a in args:
                    try: nums.append(float(self._eval_expr(a, row, col)))
                    except: pass
                return sum(nums) / len(nums) if nums else "#DIV/0!"

        elif name == "MIN":
            if range_match:
                vals = self._parse_numeric(self.get_range_values(args_str.strip()))
                return min(vals) if vals else 0
            args = self._split_args(args_str)
            nums = []
            for a in args:
                try: nums.append(float(self._eval_expr(a, row, col)))
                except: pass
            return min(nums) if nums else 0

        elif name == "MAX":
            if range_match:
                vals = self._parse_numeric(self.get_range_values(args_str.strip()))
                return max(vals) if vals else 0
            args = self._split_args(args_str)
            nums = []
            for a in args:
                try: nums.append(float(self._eval_expr(a, row, col)))
                except: pass
            return max(nums) if nums else 0

        elif name == "COUNT":
            if range_match:
                vals = self.get_range_values(args_str.strip())
                return len(self._parse_numeric(vals))
            return 0

        elif name == "COUNTA":
            if range_match:
                vals = self.get_range_values(args_str.strip())
                return sum(1 for v in vals if v is not None and v != "")
            return 0

        elif name == "IF":
            args = self._split_args(args_str)
            if len(args) < 2:
                return "#ERR!"
            cond = self._eval_expr(args[0], row, col)
            try:
                cond_bool = bool(cond) if not isinstance(cond, str) else cond not in ("", "0", "False", "FALSE")
            except:
                cond_bool = False
            if cond_bool:
                return self._eval_expr(args[1], row, col)
            elif len(args) > 2:
                return self._eval_expr(args[2], row, col)
            return ""

        elif name in ("AND", "OR", "NOT"):
            args = self._split_args(args_str)
            bools = []
            for a in args:
                v = self._eval_expr(a, row, col)
                try: bools.append(bool(v))
                except: bools.append(False)
            if name == "AND":
                return all(bools)
            elif name == "OR":
                return any(bools)
            elif name == "NOT":
                return not bools[0] if bools else True

        elif name == "CONCATENATE":
            args = self._split_args(args_str)
            parts = []
            for a in args:
                v = self._eval_expr(a, row, col)
                parts.append(str(v) if v is not None else "")
            return "".join(parts)

        elif name == "LEN":
            v = self._eval_expr(args_str.strip(), row, col)
            return len(str(v)) if v is not None else 0

        elif name in ("LEFT", "RIGHT", "MID"):
            args = self._split_args(args_str)
            text = str(self._eval_expr(args[0], row, col))
            if name == "LEFT":
                n = int(self._eval_expr(args[1], row, col)) if len(args) > 1 else 1
                return text[:n]
            elif name == "RIGHT":
                n = int(self._eval_expr(args[1], row, col)) if len(args) > 1 else 1
                return text[-n:] if n > 0 else ""
            else:  # MID
                start = int(self._eval_expr(args[1], row, col)) - 1 if len(args) > 1 else 0
                n = int(self._eval_expr(args[2], row, col)) if len(args) > 2 else 1
                return text[start:start + n]

        elif name == "UPPER":
            return str(self._eval_expr(args_str.strip(), row, col)).upper()
        elif name == "LOWER":
            return str(self._eval_expr(args_str.strip(), row, col)).lower()
        elif name == "TRIM":
            return str(self._eval_expr(args_str.strip(), row, col)).strip()

        elif name == "ROUND":
            args = self._split_args(args_str)
            v = float(self._eval_expr(args[0], row, col))
            decimals = int(self._eval_expr(args[1], row, col)) if len(args) > 1 else 0
            return round(v, decimals)

        elif name == "ABS":
            return abs(float(self._eval_expr(args_str.strip(), row, col)))
        elif name == "POWER":
            args = self._split_args(args_str)
            return float(self._eval_expr(args[0], row, col)) ** float(self._eval_expr(args[1], row, col))
        elif name == "SQRT":
            return math.sqrt(float(self._eval_expr(args_str.strip(), row, col)))
        elif name == "MOD":
            args = self._split_args(args_str)
            return float(self._eval_expr(args[0], row, col)) % float(self._eval_expr(args[1], row, col))
        elif name == "INT":
            return int(float(self._eval_expr(args_str.strip(), row, col)))

        elif name == "PI":
            return math.pi
        elif name == "RAND":
            import random
            return random.random()

        elif name == "NOW":
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elif name == "TODAY":
            return datetime.now().strftime("%Y-%m-%d")

        elif name == "ISNUMBER":
            v = self._eval_expr(args_str.strip(), row, col)
            try:
                float(v)
                return True
            except:
                return False
        elif name == "ISTEXT":
            v = self._eval_expr(args_str.strip(), row, col)
            return isinstance(v, str) and not v.replace('.','',1).replace('-','',1).isdigit()
        elif name == "ISBLANK":
            v = self._eval_expr(args_str.strip(), row, col)
            return v is None or v == ""

        elif name == "IFERROR":
            args = self._split_args(args_str)
            try:
                v = self._eval_expr(args[0], row, col)
                if isinstance(v, str) and v.startswith("#"):
                    return self._eval_expr(args[1], row, col) if len(args) > 1 else ""
                return v
            except:
                return self._eval_expr(args[1], row, col) if len(args) > 1 else ""

        elif name == "SUMIF":
            args = self._split_args(args_str)
            if len(args) < 2:
                return "#ERR!"
            range_vals = self.get_range_values(args[0])
            criteria = self._eval_expr(args[1], row, col)
            sum_range = self.get_range_values(args[2]) if len(args) > 2 else range_vals
            total = 0
            for i, v in enumerate(range_vals):
                if self._check_criteria(v, criteria):
                    if i < len(sum_range):
                        try: total += float(sum_range[i])
                        except: pass
            return total

        elif name == "COUNTIF":
            args = self._split_args(args_str)
            if len(args) < 2:
                return "#ERR!"
            range_vals = self.get_range_values(args[0])
            criteria = self._eval_expr(args[1], row, col)
            count = 0
            for v in range_vals:
                if self._check_criteria(v, criteria):
                    count += 1
            return count

        elif name == "VLOOKUP":
            args = self._split_args(args_str)
            if len(args) < 3:
                return "#ERR!"
            lookup = self._eval_expr(args[0], row, col)
            range_vals = args[1].strip()
            col_idx = int(self._eval_expr(args[2], row, col)) - 1
            rm = self.RANGE_RE.match(range_vals)
            if not rm:
                return "#ERR!"
            c1, r1, c2, r2 = rm.groups()
            col1, col2 = self.col_to_index(c1), self.col_to_index(c2)
            row1, row2 = int(r1) - 1, int(r2) - 1
            for r in range(row1, row2 + 1):
                v = self.get_cell(r, col1)
                try:
                    if str(v) == str(lookup) or float(v) == float(lookup):
                        return self.get_cell(r, col1 + col_idx)
                except:
                    if str(v) == str(lookup):
                        return self.get_cell(r, col1 + col_idx)
            return "#N/A"

        elif name == "TEXT":
            args = self._split_args(args_str)
            v = self._eval_expr(args[0], row, col)
            fmt = str(self._eval_expr(args[1], row, col)).strip('"').strip("'")
            try:
                if fmt == "0.00":
                    return f"{float(v):.2f}"
                elif fmt == "0%":
                    return f"{float(v)*100:.0f}%"
                elif fmt == "0.0%":
                    return f"{float(v)*100:.1f}%"
                elif fmt == "#,##0":
                    return f"{float(v):,.0f}"
                elif fmt == "#,##0.00":
                    return f"{float(v):,.2f}"
                elif fmt == "$#,##0.00":
                    return f"${float(v):,.2f}"
                else:
                    return str(v)
            except:
                return str(v)

        elif name == "VALUE":
            v = str(self._eval_expr(args_str.strip(), row, col))
            v = v.replace(",", "").replace("$", "").replace("%", "")
            try:
                return float(v)
            except:
                return "#VALUE!"

        return "#NAME?"

    def _check_criteria(self, value, criteria):
        """Check if a value meets criteria like ">10", "=text", "<5" etc."""
        crit_str = str(criteria)
        if crit_str.startswith(">="):
            try: return float(value) >= float(crit_str[2:])
            except: return False
        elif crit_str.startswith("<="):
            try: return float(value) <= float(crit_str[2:])
            except: return False
        elif crit_str.startswith("<>"):
            return str(value) != crit_str[2:]
        elif crit_str.startswith(">"):
            try: return float(value) > float(crit_str[1:])
            except: return False
        elif crit_str.startswith("<"):
            try: return float(value) < float(crit_str[1:])
            except: return False
        elif crit_str.startswith("="):
            return str(value) == crit_str[1:]
        else:
            try:
                return float(value) == float(crit_str)
            except:
                return str(value) == crit_str

    def _replace_cell_refs(self, expr, row, col):
        """Replace cell references like A1, $B$2 with their values"""
        # Don't replace things inside strings
        def replacer(m):
            col_str = m.group(1)
            row_num = int(m.group(2))
            c = self.col_to_index(col_str)
            r = row_num - 1
            val = self.get_cell(r, c)
            if val is None or val == "":
                return "0"
            try:
                float(val)
                return str(val)
            except:
                return f'"{val}"'

        return self.CELL_RE.sub(replacer, expr)


# ─────────────────────────────────────────────
# CELL DATA MODEL
# ─────────────────────────────────────────────
class CellData:
    """Rich cell data with formatting"""
    __slots__ = [
        'raw_value', 'display_value', 'formula',
        'font_family', 'font_size', 'bold', 'italic', 'underline',
        'text_color', 'bg_color', 'alignment',
        'number_format', 'border_style',
        'comment', 'locked', 'merged_with'
    ]

    def __init__(self, raw_value=""):
        self.raw_value = raw_value
        self.display_value = raw_value
        self.formula = None
        self.font_family = "JetBrains Mono"
        self.font_size = 11
        self.bold = False
        self.italic = False
        self.underline = False
        self.text_color = None
        self.bg_color = None
        self.alignment = Qt.AlignLeft | Qt.AlignVCenter
        self.number_format = None  # None, "$", "%", ",", "0.00" etc.
        self.border_style = None
        self.comment = None
        self.locked = False
        self.merged_with = None

    def clone(self):
        c = CellData(self.raw_value)
        for attr in self.__slots__:
            setattr(c, attr, getattr(self, attr))
        return c


# ─────────────────────────────────────────────
# SHEET DATA MODEL
# ─────────────────────────────────────────────
class SheetModel:
    """Data model for a single sheet"""

    def __init__(self, name="Sheet1", rows=200, cols=52):
        self.name = name
        self.rows = rows
        self.cols = cols
        self.cells = {}  # (row, col) -> CellData
        self.col_widths = {}
        self.row_heights = {}
        self.frozen_rows = 0
        self.frozen_cols = 0
        self.conditional_formats = []  # list of ConditionalFormat

    def get_cell(self, row, col):
        return self.cells.get((row, col))

    def set_cell(self, row, col, cell_data):
        self.cells[(row, col)] = cell_data

    def get_value(self, row, col):
        cell = self.cells.get((row, col))
        if cell is None:
            return None
        return cell.display_value if cell.display_value is not None else cell.raw_value

    def get_raw_value(self, row, col):
        cell = self.cells.get((row, col))
        return cell.raw_value if cell else None

    def clear_cell(self, row, col):
        if (row, col) in self.cells:
            del self.cells[(row, col)]

    def to_dict(self):
        """Serialize sheet to dict"""
        data = {
            'name': self.name,
            'rows': self.rows,
            'cols': self.cols,
            'frozen_rows': self.frozen_rows,
            'frozen_cols': self.frozen_cols,
            'cells': {},
            'col_widths': {str(k): v for k, v in self.col_widths.items()},
            'row_heights': {str(k): v for k, v in self.row_heights.items()},
        }
        for (r, c), cell in self.cells.items():
            key = f"{r},{c}"
            cell_dict = {'raw': cell.raw_value}
            if cell.formula:
                cell_dict['formula'] = cell.formula
            if cell.bold: cell_dict['bold'] = True
            if cell.italic: cell_dict['italic'] = True
            if cell.underline: cell_dict['underline'] = True
            if cell.text_color: cell_dict['text_color'] = cell.text_color
            if cell.bg_color: cell_dict['bg_color'] = cell.bg_color
            if cell.font_size != 11: cell_dict['font_size'] = cell.font_size
            if cell.number_format: cell_dict['number_format'] = cell.number_format
            if cell.comment: cell_dict['comment'] = cell.comment
            data['cells'][key] = cell_dict
        return data


class ConditionalFormat:
    """Conditional formatting rule"""
    def __init__(self, range_str, rule_type, value, format_dict):
        self.range_str = range_str  # e.g. "A1:A100"
        self.rule_type = rule_type  # "greater_than", "less_than", "equal", "contains", "between", "color_scale"
        self.value = value
        self.format_dict = format_dict  # {'bg_color': '#...', 'text_color': '#...'}


# ─────────────────────────────────────────────
# UNDO/REDO SYSTEM
# ─────────────────────────────────────────────
class UndoManager:
    def __init__(self, max_history=100):
        self.undo_stack = []
        self.redo_stack = []
        self.max_history = max_history

    def push(self, action):
        """action = {'type': str, 'data': any_before_state, 'redo_data': any_after_state}"""
        self.undo_stack.append(action)
        if len(self.undo_stack) > self.max_history:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def can_undo(self):
        return len(self.undo_stack) > 0

    def can_redo(self):
        return len(self.redo_stack) > 0

    def undo(self):
        if self.undo_stack:
            action = self.undo_stack.pop()
            self.redo_stack.append(action)
            return action
        return None

    def redo(self):
        if self.redo_stack:
            action = self.redo_stack.pop()
            self.undo_stack.append(action)
            return action
        return None


# ─────────────────────────────────────────────
# SPARKLINE DELEGATE (mini charts in cells)
# ─────────────────────────────────────────────
class SparklineDelegate(QStyledItemDelegate):
    """Custom delegate for rendering sparklines in cells"""
    def __init__(self, parent=None):
        super().__init__(parent)

    def paint(self, painter, option, index):
        # Default paint first
        super().paint(painter, option, index)


# ─────────────────────────────────────────────
# MINI CHART WIDGET (for embedded charts)
# ─────────────────────────────────────────────
class MiniChartWidget(QWidget):
    """Embeddable chart widget for the spreadsheet"""

    def __init__(self, chart_type="bar", data=None, labels=None, title="", parent=None):
        super().__init__(parent)
        self.chart_type = chart_type  # "bar", "line", "pie", "area"
        self.data = data or []
        self.labels = labels or []
        self.title = title
        self.colors = [QColor(c) for c in Theme.CHART_COLORS]
        self.setMinimumSize(300, 200)
        self.hover_idx = -1
        self.setMouseTracking(True)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = self.rect().adjusted(10, 10, -10, -10)

        # Background
        painter.setBrush(QBrush(QColor(Theme.BG_SECONDARY)))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(self.rect(), 8, 8)

        if not self.data:
            painter.setPen(QPen(QColor(Theme.TEXT_DIM)))
            painter.drawText(rect, Qt.AlignCenter, "No data")
            painter.end()
            return

        # Title
        if self.title:
            painter.setPen(QPen(QColor(Theme.TEXT_PRIMARY)))
            title_font = QFont("Segoe UI", 10, QFont.Bold)
            painter.setFont(title_font)
            painter.drawText(rect.adjusted(5, 0, 0, 0), Qt.AlignTop | Qt.AlignLeft, self.title)
            rect.adjust(0, 25, 0, 0)

        chart_rect = rect.adjusted(5, 5, -5, -25)

        if self.chart_type == "bar":
            self._draw_bar_chart(painter, chart_rect)
        elif self.chart_type == "line":
            self._draw_line_chart(painter, chart_rect)
        elif self.chart_type == "pie":
            self._draw_pie_chart(painter, chart_rect)
        elif self.chart_type == "area":
            self._draw_area_chart(painter, chart_rect)

        painter.end()

    def _draw_bar_chart(self, painter, rect):
        if not self.data:
            return
        max_val = max(abs(v) for v in self.data) if self.data else 1
        if max_val == 0:
            max_val = 1
        n = len(self.data)
        bar_w = max(4, (rect.width() - (n - 1) * 3) // n)
        baseline_y = rect.bottom()

        for i, val in enumerate(self.data):
            x = rect.left() + i * (bar_w + 3)
            h = int((abs(val) / max_val) * rect.height() * 0.85)
            color = self.colors[i % len(self.colors)]
            if i == self.hover_idx:
                color = color.lighter(130)

            # Gradient fill
            grad = QLinearGradient(x, baseline_y - h, x, baseline_y)
            grad.setColorAt(0, color)
            grad.setColorAt(1, color.darker(140))
            painter.setBrush(QBrush(grad))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(x, baseline_y - h, bar_w, h, 3, 3)

            # Value label
            painter.setPen(QPen(QColor(Theme.TEXT_SECONDARY)))
            painter.setFont(QFont("JetBrains Mono", 7))
            label = f"{val:g}" if isinstance(val, float) else str(val)
            painter.drawText(QRect(x, baseline_y - h - 15, bar_w, 14),
                           Qt.AlignCenter, label)

    def _draw_line_chart(self, painter, rect):
        if len(self.data) < 2:
            return
        max_val = max(self.data)
        min_val = min(self.data)
        val_range = max_val - min_val if max_val != min_val else 1
        n = len(self.data)
        step_x = rect.width() / (n - 1) if n > 1 else rect.width()

        points = []
        for i, val in enumerate(self.data):
            x = rect.left() + i * step_x
            y = rect.bottom() - ((val - min_val) / val_range) * rect.height() * 0.85
            points.append(QPoint(int(x), int(y)))

        # Line
        pen = QPen(QColor(Theme.ACCENT), 2.5)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(pen)
        for i in range(len(points) - 1):
            painter.drawLine(points[i], points[i + 1])

        # Dots
        painter.setBrush(QBrush(QColor(Theme.ACCENT)))
        for i, p in enumerate(points):
            r = 5 if i == self.hover_idx else 3
            painter.drawEllipse(p, r, r)

    def _draw_area_chart(self, painter, rect):
        if len(self.data) < 2:
            return
        max_val = max(self.data)
        min_val = min(self.data)
        val_range = max_val - min_val if max_val != min_val else 1
        n = len(self.data)
        step_x = rect.width() / (n - 1) if n > 1 else rect.width()

        from PySide6.QtGui import QPolygonF
        from PySide6.QtCore import QPointF

        polygon = QPolygonF()
        polygon.append(QPointF(rect.left(), rect.bottom()))

        for i, val in enumerate(self.data):
            x = rect.left() + i * step_x
            y = rect.bottom() - ((val - min_val) / val_range) * rect.height() * 0.85
            polygon.append(QPointF(x, y))

        polygon.append(QPointF(rect.right(), rect.bottom()))

        grad = QLinearGradient(0, rect.top(), 0, rect.bottom())
        color = QColor(Theme.ACCENT)
        color.setAlpha(120)
        grad.setColorAt(0, color)
        color.setAlpha(20)
        grad.setColorAt(1, color)
        painter.setBrush(QBrush(grad))
        painter.setPen(Qt.NoPen)
        painter.drawPolygon(polygon)

        # Line on top
        pen = QPen(QColor(Theme.ACCENT), 2)
        painter.setPen(pen)
        points = []
        for i, val in enumerate(self.data):
            x = rect.left() + i * step_x
            y = rect.bottom() - ((val - min_val) / val_range) * rect.height() * 0.85
            points.append(QPoint(int(x), int(y)))
        for i in range(len(points) - 1):
            painter.drawLine(points[i], points[i + 1])

    def _draw_pie_chart(self, painter, rect):
        if not self.data:
            return
        total = sum(abs(v) for v in self.data)
        if total == 0:
            return

        size = min(rect.width(), rect.height())
        pie_rect = QRect(
            rect.center().x() - size // 2,
            rect.center().y() - size // 2,
            size, size
        )
        pie_rect.adjust(10, 10, -10, -10)

        start_angle = 90 * 16
        for i, val in enumerate(self.data):
            span = int((abs(val) / total) * 360 * 16)
            color = self.colors[i % len(self.colors)]
            if i == self.hover_idx:
                color = color.lighter(130)
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(QColor(Theme.BG_SECONDARY), 2))
            painter.drawPie(pie_rect, start_angle, span)
            start_angle += span

    def mouseMoveEvent(self, event):
        # Simple hover detection for bars
        if self.chart_type == "bar" and self.data:
            rect = self.rect().adjusted(15, 35, -15, -30)
            n = len(self.data)
            bar_w = max(4, (rect.width() - (n - 1) * 3) // n)
            x = event.pos().x() - rect.left()
            idx = int(x / (bar_w + 3))
            if 0 <= idx < n:
                self.hover_idx = idx
            else:
                self.hover_idx = -1
            self.update()


# ─────────────────────────────────────────────
# CHART DIALOG
# ─────────────────────────────────────────────
class ChartDialog(QDialog):
    """Dialog for creating charts from selected data"""

    def __init__(self, data, labels, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Insert Chart")
        self.setMinimumSize(500, 500)
        self.data = data
        self.labels = labels
        self.chart_type = "bar"

        self.setStyleSheet(f"""
            QDialog {{
                background: {Theme.BG_PRIMARY};
                color: {Theme.TEXT_PRIMARY};
            }}
            QPushButton {{
                background: {Theme.BG_TERTIARY};
                color: {Theme.TEXT_PRIMARY};
                border: 1px solid {Theme.BORDER};
                padding: 8px 16px;
                border-radius: 6px;
                font-size: 12px;
            }}
            QPushButton:hover {{
                background: {Theme.BG_HOVER};
                border-color: {Theme.ACCENT};
            }}
            QPushButton:checked {{
                background: {Theme.ACCENT};
                color: #FFFFFF;
            }}
            QLineEdit {{
                background: {Theme.BG_INPUT};
                color: {Theme.TEXT_PRIMARY};
                border: 1px solid {Theme.BORDER};
                padding: 6px;
                border-radius: 4px;
            }}
            QLabel {{
                color: {Theme.TEXT_SECONDARY};
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Title input
        title_row = QHBoxLayout()
        title_row.addWidget(QLabel("Chart Title:"))
        self.title_edit = QLineEdit()
        self.title_edit.setPlaceholderText("My Chart")
        title_row.addWidget(self.title_edit)
        layout.addLayout(title_row)

        # Chart type selector
        type_row = QHBoxLayout()
        self.type_buttons = {}
        for ctype in ["bar", "line", "area", "pie"]:
            btn = QPushButton(ctype.title())
            btn.setCheckable(True)
            btn.setChecked(ctype == "bar")
            btn.clicked.connect(lambda checked, t=ctype: self._set_type(t))
            self.type_buttons[ctype] = btn
            type_row.addWidget(btn)
        layout.addLayout(type_row)

        # Preview
        self.preview = MiniChartWidget("bar", data, labels, "Preview")
        self.preview.setMinimumHeight(250)
        layout.addWidget(self.preview)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)
        insert_btn = QPushButton("Insert Chart")
        insert_btn.setStyleSheet(f"background: {Theme.ACCENT}; color: #FFFFFF; font-weight: bold;")
        insert_btn.clicked.connect(self.accept)
        btn_row.addWidget(insert_btn)
        layout.addLayout(btn_row)

    def _set_type(self, chart_type):
        self.chart_type = chart_type
        for t, btn in self.type_buttons.items():
            btn.setChecked(t == chart_type)
        self.preview.chart_type = chart_type
        self.preview.update()

    def get_result(self):
        return {
            'type': self.chart_type,
            'title': self.title_edit.text() or "Chart",
            'data': self.data,
            'labels': self.labels
        }


# ─────────────────────────────────────────────
# FIND & REPLACE DIALOG
# ─────────────────────────────────────────────
class FindReplaceDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Find & Replace")
        self.setMinimumWidth(400)
        self.setStyleSheet(f"""
            QDialog {{ background: {Theme.BG_PRIMARY}; color: {Theme.TEXT_PRIMARY}; }}
            QLineEdit {{
                background: {Theme.BG_INPUT}; color: {Theme.TEXT_PRIMARY};
                border: 1px solid {Theme.BORDER}; padding: 8px; border-radius: 4px;
                font-size: 13px;
            }}
            QPushButton {{
                background: {Theme.BG_TERTIARY}; color: {Theme.TEXT_PRIMARY};
                border: 1px solid {Theme.BORDER}; padding: 8px 16px; border-radius: 6px;
            }}
            QPushButton:hover {{ background: {Theme.BG_HOVER}; border-color: {Theme.ACCENT}; }}
            QCheckBox {{ color: {Theme.TEXT_SECONDARY}; }}
            QLabel {{ color: {Theme.TEXT_SECONDARY}; }}
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        find_row = QHBoxLayout()
        find_row.addWidget(QLabel("Find:"))
        self.find_edit = QLineEdit()
        find_row.addWidget(self.find_edit)
        layout.addLayout(find_row)

        replace_row = QHBoxLayout()
        replace_row.addWidget(QLabel("Replace:"))
        self.replace_edit = QLineEdit()
        replace_row.addWidget(self.replace_edit)
        layout.addLayout(replace_row)

        opts = QHBoxLayout()
        self.case_check = QCheckBox("Case sensitive")
        self.whole_check = QCheckBox("Whole cell")
        opts.addWidget(self.case_check)
        opts.addWidget(self.whole_check)
        opts.addStretch()
        layout.addLayout(opts)

        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        btns = QHBoxLayout()
        self.find_btn = QPushButton("Find Next")
        self.replace_btn = QPushButton("Replace")
        self.replace_all_btn = QPushButton("Replace All")
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        btns.addWidget(self.find_btn)
        btns.addWidget(self.replace_btn)
        btns.addWidget(self.replace_all_btn)
        btns.addStretch()
        btns.addWidget(close_btn)
        layout.addLayout(btns)


# ─────────────────────────────────────────────
# CONDITIONAL FORMAT DIALOG
# ─────────────────────────────────────────────
class ConditionalFormatDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Conditional Formatting")
        self.setMinimumWidth(450)
        self.setStyleSheet(f"""
            QDialog {{ background: {Theme.BG_PRIMARY}; color: {Theme.TEXT_PRIMARY}; }}
            QComboBox, QLineEdit, QSpinBox {{
                background: {Theme.BG_INPUT}; color: {Theme.TEXT_PRIMARY};
                border: 1px solid {Theme.BORDER}; padding: 6px; border-radius: 4px;
            }}
            QPushButton {{
                background: {Theme.BG_TERTIARY}; color: {Theme.TEXT_PRIMARY};
                border: 1px solid {Theme.BORDER}; padding: 8px 16px; border-radius: 6px;
            }}
            QPushButton:hover {{ background: {Theme.BG_HOVER}; border-color: {Theme.ACCENT}; }}
            QLabel {{ color: {Theme.TEXT_SECONDARY}; }}
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Rule type
        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Rule:"))
        self.rule_combo = QComboBox()
        self.rule_combo.addItems([
            "Greater than", "Less than", "Equal to",
            "Between", "Contains text", "Color scale"
        ])
        type_row.addWidget(self.rule_combo)
        layout.addLayout(type_row)

        # Value
        val_row = QHBoxLayout()
        val_row.addWidget(QLabel("Value:"))
        self.value_edit = QLineEdit()
        val_row.addWidget(self.value_edit)
        layout.addLayout(val_row)

        # Format
        fmt_row = QHBoxLayout()
        fmt_row.addWidget(QLabel("Format:"))
        self.bg_color_btn = QPushButton("Background")
        self.bg_color_btn.clicked.connect(self._pick_bg_color)
        self.text_color_btn = QPushButton("Text Color")
        self.text_color_btn.clicked.connect(self._pick_text_color)
        fmt_row.addWidget(self.bg_color_btn)
        fmt_row.addWidget(self.text_color_btn)
        layout.addLayout(fmt_row)

        self.chosen_bg = Theme.DANGER
        self.chosen_text = Theme.TEXT_PRIMARY

        # Buttons
        btns = QHBoxLayout()
        btns.addStretch()
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        apply_btn = QPushButton("Apply")
        apply_btn.setStyleSheet(f"background: {Theme.ACCENT}; color: #FFFFFF; font-weight: bold;")
        apply_btn.clicked.connect(self.accept)
        btns.addWidget(cancel)
        btns.addWidget(apply_btn)
        layout.addLayout(btns)

    def _pick_bg_color(self):
        c = QColorDialog.getColor(QColor(self.chosen_bg), self)
        if c.isValid():
            self.chosen_bg = c.name()
            self.bg_color_btn.setStyleSheet(f"background: {c.name()};")

    def _pick_text_color(self):
        c = QColorDialog.getColor(QColor(self.chosen_text), self)
        if c.isValid():
            self.chosen_text = c.name()
            self.text_color_btn.setStyleSheet(f"background: {c.name()};")

    def get_result(self):
        rule_map = {
            "Greater than": "greater_than",
            "Less than": "less_than",
            "Equal to": "equal",
            "Between": "between",
            "Contains text": "contains",
            "Color scale": "color_scale",
        }
        return {
            'rule_type': rule_map.get(self.rule_combo.currentText(), "greater_than"),
            'value': self.value_edit.text(),
            'bg_color': self.chosen_bg,
            'text_color': self.chosen_text,
        }


# ─────────────────────────────────────────────
# MAIN GRIDION WIDGET
# ─────────────────────────────────────────────
class Gridion(QWidget):
    """
    GRIDION — Next-generation spreadsheet engine.
    Features beyond Excel:
     ✦ Dark-mode-first design with Midnight Aurora theme
     ✦ Full formula engine with 50+ functions
     ✦ Multi-sheet tabs with drag reordering
     ✦ Embedded charts (bar, line, area, pie)
     ✦ Conditional formatting with color scales
     ✦ Undo/Redo with 100-step history
     ✦ Find & Replace with regex
     ✦ CSV/JSON import and export
     ✦ Cell comments and annotations
     ✦ Freeze panes
     ✦ Auto-column-resize
     ✦ Drag-fill for series (numbers, dates)
     ✦ Real-time formula bar with syntax highlighting
     ✦ Status bar with live SUM, AVG, COUNT, MIN, MAX
     ✦ Cell formatting: bold, italic, underline, colors, alignment, number formats
     ✦ Context menu with rich options
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("GRIDION")
        self.setMinimumSize(1100, 720)

        # State
        self.sheets = [SheetModel("Sheet 1")]
        self.current_sheet_idx = 0
        self.undo_mgr = UndoManager()
        self.clipboard = []  # list of (rel_row, rel_col, CellData)
        self.selection_anchor = None
        self.charts = []  # embedded chart widgets

        # Formula engine
        self.formula_engine = FormulaEngine(self._get_cell_value_for_formula)

        self._build_ui()
        self._apply_global_style()
        self._connect_signals()
        self._populate_table()

    # ═══════════════════════════════════════
    # UI CONSTRUCTION
    # ═══════════════════════════════════════
    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ── Menu Bar ──
        self.menu_bar = QMenuBar()
        self._build_menus()
        main_layout.addWidget(self.menu_bar)

        # ── Toolbar ──
        self.toolbar = QFrame()
        self.toolbar.setFixedHeight(44)
        self.toolbar.setObjectName("toolbar")
        toolbar_layout = QHBoxLayout(self.toolbar)
        toolbar_layout.setContentsMargins(8, 4, 8, 4)
        toolbar_layout.setSpacing(4)
        self._build_toolbar(toolbar_layout)
        main_layout.addWidget(self.toolbar)

        # ── Formatting Bar ──
        self.format_bar = QFrame()
        self.format_bar.setFixedHeight(40)
        self.format_bar.setObjectName("formatBar")
        fmt_layout = QHBoxLayout(self.format_bar)
        fmt_layout.setContentsMargins(8, 2, 8, 2)
        fmt_layout.setSpacing(6)
        self._build_format_bar(fmt_layout)
        main_layout.addWidget(self.format_bar)

        # ── Formula Bar ──
        self.formula_frame = QFrame()
        self.formula_frame.setFixedHeight(36)
        self.formula_frame.setObjectName("formulaBar")
        formula_layout = QHBoxLayout(self.formula_frame)
        formula_layout.setContentsMargins(8, 2, 8, 2)
        formula_layout.setSpacing(6)

        self.cell_ref_label = QLabel("A1")
        self.cell_ref_label.setFixedWidth(70)
        self.cell_ref_label.setAlignment(Qt.AlignCenter)
        self.cell_ref_label.setObjectName("cellRef")
        formula_layout.addWidget(self.cell_ref_label)

        fx_label = QLabel("𝑓𝑥")
        fx_label.setFixedWidth(24)
        fx_label.setAlignment(Qt.AlignCenter)
        fx_label.setStyleSheet(f"color: {Theme.ACCENT}; font-size: 14px; font-weight: bold;")
        formula_layout.addWidget(fx_label)

        self.formula_edit = QLineEdit()
        self.formula_edit.setObjectName("formulaEdit")
        self.formula_edit.setPlaceholderText("Enter value or formula (e.g. =SUM(A1:A10))")
        formula_layout.addWidget(self.formula_edit)
        main_layout.addWidget(self.formula_frame)

        # ── Separator ──
        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background: {Theme.BORDER};")
        main_layout.addWidget(sep)

        # ── Table ──
        self.table = QTableWidget()
        self.table.setObjectName("mainGrid")
        self._configure_table()
        main_layout.addWidget(self.table, 1)

        # ── Sheet Tabs ──
        self.tab_bar_frame = QFrame()
        self.tab_bar_frame.setFixedHeight(34)
        self.tab_bar_frame.setObjectName("tabFrame")
        tab_layout = QHBoxLayout(self.tab_bar_frame)
        tab_layout.setContentsMargins(4, 2, 4, 2)
        tab_layout.setSpacing(4)

        self.add_sheet_btn = QPushButton("+")
        self.add_sheet_btn.setFixedSize(28, 26)
        self.add_sheet_btn.setObjectName("addSheetBtn")
        self.add_sheet_btn.setToolTip("Add sheet")
        tab_layout.addWidget(self.add_sheet_btn)

        self.sheet_tab_bar = QTabBar()
        self.sheet_tab_bar.setObjectName("sheetTabs")
        self.sheet_tab_bar.setExpanding(False)
        self.sheet_tab_bar.setMovable(True)
        self.sheet_tab_bar.addTab("Sheet 1")
        tab_layout.addWidget(self.sheet_tab_bar)
        tab_layout.addStretch()

        main_layout.addWidget(self.tab_bar_frame)

        # ── Status Bar ──
        self.status_frame = QFrame()
        self.status_frame.setFixedHeight(28)
        self.status_frame.setObjectName("statusFrame")
        status_layout = QHBoxLayout(self.status_frame)
        status_layout.setContentsMargins(12, 0, 12, 0)

        self.status_left = QLabel("Ready")
        self.status_left.setObjectName("statusLeft")
        status_layout.addWidget(self.status_left)
        status_layout.addStretch()

        self.status_stats = QLabel("")
        self.status_stats.setObjectName("statusStats")
        status_layout.addWidget(self.status_stats)

        main_layout.addWidget(self.status_frame)

    def _build_menus(self):
        # File menu
        file_menu = self.menu_bar.addMenu("File")
        self._add_action(file_menu, "New", "Ctrl+N", self._new_workbook)
        file_menu.addSeparator()
        self._add_action(file_menu, "Import CSV...", "Ctrl+O", self._import_csv)
        self._add_action(file_menu, "Import JSON...", "", self._import_json)
        file_menu.addSeparator()
        self._add_action(file_menu, "Export CSV...", "Ctrl+Shift+S", self._export_csv)
        self._add_action(file_menu, "Export JSON...", "", self._export_json)

        # Edit menu
        edit_menu = self.menu_bar.addMenu("Edit")
        self._add_action(edit_menu, "Undo", "Ctrl+Z", self._undo)
        self._add_action(edit_menu, "Redo", "Ctrl+Y", self._redo)
        edit_menu.addSeparator()
        self._add_action(edit_menu, "Cut", "Ctrl+X", self._cut)
        self._add_action(edit_menu, "Copy", "Ctrl+C", self._copy)
        self._add_action(edit_menu, "Paste", "Ctrl+V", self._paste)
        self._add_action(edit_menu, "Delete", "Delete", self._delete_selection)
        edit_menu.addSeparator()
        self._add_action(edit_menu, "Find && Replace...", "Ctrl+H", self._show_find_replace)
        self._add_action(edit_menu, "Select All", "Ctrl+A", self._select_all)

        # Insert menu
        insert_menu = self.menu_bar.addMenu("Insert")
        self._add_action(insert_menu, "Chart...", "", self._insert_chart)
        self._add_action(insert_menu, "Row Above", "", self._insert_row_above)
        self._add_action(insert_menu, "Row Below", "", self._insert_row_below)
        self._add_action(insert_menu, "Column Left", "", self._insert_col_left)
        self._add_action(insert_menu, "Column Right", "", self._insert_col_right)
        insert_menu.addSeparator()
        self._add_action(insert_menu, "Comment", "", self._add_comment)

        # Format menu
        format_menu = self.menu_bar.addMenu("Format")
        self._add_action(format_menu, "Bold", "Ctrl+B", self._toggle_bold)
        self._add_action(format_menu, "Italic", "Ctrl+I", self._toggle_italic)
        self._add_action(format_menu, "Underline", "Ctrl+U", self._toggle_underline)
        format_menu.addSeparator()
        self._add_action(format_menu, "Text Color...", "", self._pick_text_color)
        self._add_action(format_menu, "Cell Color...", "", self._pick_bg_color)
        format_menu.addSeparator()
        self._add_action(format_menu, "Conditional Formatting...", "", self._show_cond_format)
        format_menu.addSeparator()
        self._add_action(format_menu, "Auto-fit Column Width", "", self._auto_fit_columns)

        # View menu
        view_menu = self.menu_bar.addMenu("View")
        self._add_action(view_menu, "Freeze Panes", "", self._toggle_freeze)

        # Data menu
        data_menu = self.menu_bar.addMenu("Data")
        self._add_action(data_menu, "Sort A→Z", "", lambda: self._sort_column(True))
        self._add_action(data_menu, "Sort Z→A", "", lambda: self._sort_column(False))
        data_menu.addSeparator()
        self._add_action(data_menu, "Fill Series Down", "Ctrl+D", self._fill_down)

    def _add_action(self, menu, text, shortcut, callback):
        action = QAction(text, self)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        action.triggered.connect(callback)
        menu.addAction(action)
        return action

    def _build_toolbar(self, layout):
        """Build main toolbar with common actions"""
        btn_style = "tb_btn"

        # Undo/Redo
        self.undo_btn = self._make_tb_btn("↶", "Undo (Ctrl+Z)", self._undo)
        self.redo_btn = self._make_tb_btn("↷", "Redo (Ctrl+Y)", self._redo)
        layout.addWidget(self.undo_btn)
        layout.addWidget(self.redo_btn)

        layout.addWidget(self._make_sep())

        # Cut/Copy/Paste
        layout.addWidget(self._make_tb_btn("✂", "Cut", self._cut))
        layout.addWidget(self._make_tb_btn("⧉", "Copy", self._copy))
        layout.addWidget(self._make_tb_btn("⎘", "Paste", self._paste))

        layout.addWidget(self._make_sep())

        # Import/Export
        layout.addWidget(self._make_tb_btn("⬆", "Import CSV", self._import_csv))
        layout.addWidget(self._make_tb_btn("⬇", "Export CSV", self._export_csv))

        layout.addWidget(self._make_sep())

        # Chart
        layout.addWidget(self._make_tb_btn("📊", "Insert Chart", self._insert_chart))

        # Find
        layout.addWidget(self._make_tb_btn("🔍", "Find & Replace", self._show_find_replace))

        layout.addStretch()

        # Zoom indicator
        self.zoom_label = QLabel("100%")
        self.zoom_label.setObjectName("zoomLabel")
        layout.addWidget(self.zoom_label)

    def _build_format_bar(self, layout):
        """Build formatting toolbar"""
        # Font family
        self.font_combo = QComboBox()
        self.font_combo.setFixedWidth(140)
        self.font_combo.setObjectName("fontCombo")
        self.font_combo.addItems([
            "JetBrains Mono", "Fira Code", "Cascadia Code", "Source Code Pro",
            "Consolas", "Menlo", "Monaco", "Courier New",
            "Segoe UI", "Helvetica", "Arial", "Georgia", "Times New Roman"
        ])
        layout.addWidget(self.font_combo)

        # Font size
        self.size_spin = QSpinBox()
        self.size_spin.setRange(6, 72)
        self.size_spin.setValue(11)
        self.size_spin.setFixedWidth(55)
        self.size_spin.setObjectName("sizeSpin")
        layout.addWidget(self.size_spin)

        layout.addWidget(self._make_sep())

        # Bold / Italic / Underline
        self.bold_btn = self._make_fmt_btn("B", "Bold (Ctrl+B)", self._toggle_bold, bold=True)
        self.italic_btn = self._make_fmt_btn("I", "Italic (Ctrl+I)", self._toggle_italic, italic=True)
        self.underline_btn = self._make_fmt_btn("U", "Underline (Ctrl+U)", self._toggle_underline, underline=True)
        layout.addWidget(self.bold_btn)
        layout.addWidget(self.italic_btn)
        layout.addWidget(self.underline_btn)

        layout.addWidget(self._make_sep())

        # Text color
        self.text_color_btn = QPushButton("A")
        self.text_color_btn.setFixedSize(30, 28)
        self.text_color_btn.setObjectName("textColorBtn")
        self.text_color_btn.setToolTip("Text Color")
        self.text_color_btn.setStyleSheet(f"""
            QPushButton {{
                font-weight: bold; font-size: 13px;
                border-bottom: 3px solid {Theme.DANGER};
                background: transparent; color: {Theme.TEXT_PRIMARY};
                border-radius: 4px;
            }}
            QPushButton:hover {{ background: {Theme.BG_HOVER}; }}
        """)
        self.text_color_btn.clicked.connect(self._pick_text_color)
        layout.addWidget(self.text_color_btn)

        # BG color
        self.bg_color_btn = QPushButton("⬛")
        self.bg_color_btn.setFixedSize(30, 28)
        self.bg_color_btn.setToolTip("Cell Background")
        self.bg_color_btn.setStyleSheet(f"""
            QPushButton {{
                font-size: 12px; background: transparent;
                color: {Theme.WARNING}; border-radius: 4px;
            }}
            QPushButton:hover {{ background: {Theme.BG_HOVER}; }}
        """)
        self.bg_color_btn.clicked.connect(self._pick_bg_color)
        layout.addWidget(self.bg_color_btn)

        layout.addWidget(self._make_sep())

        # Alignment
        self.align_left_btn = self._make_fmt_btn("≡", "Align Left", lambda: self._set_alignment("left"))
        self.align_center_btn = self._make_fmt_btn("≡", "Align Center", lambda: self._set_alignment("center"))
        self.align_right_btn = self._make_fmt_btn("≡", "Align Right", lambda: self._set_alignment("right"))
        # Style the alignment text slightly differently
        self.align_center_btn.setStyleSheet(self.align_center_btn.styleSheet().replace("text-align: left", "text-align: center"))
        layout.addWidget(self.align_left_btn)
        layout.addWidget(self.align_center_btn)
        layout.addWidget(self.align_right_btn)

        layout.addWidget(self._make_sep())

        # Number formats
        self.num_format_combo = QComboBox()
        self.num_format_combo.setFixedWidth(100)
        self.num_format_combo.setObjectName("numFmtCombo")
        self.num_format_combo.addItems([
            "General", "$#,##0.00", "#,##0", "0.00",
            "0%", "0.0%", "Scientific"
        ])
        self.num_format_combo.currentTextChanged.connect(self._apply_number_format)
        layout.addWidget(self.num_format_combo)

        layout.addStretch()

    def _make_tb_btn(self, text, tooltip, callback):
        btn = QPushButton(text)
        btn.setFixedSize(32, 32)
        btn.setToolTip(tooltip)
        btn.setObjectName("tbBtn")
        btn.clicked.connect(callback)
        return btn

    def _make_fmt_btn(self, text, tooltip, callback, bold=False, italic=False, underline=False):
        btn = QPushButton(text)
        btn.setFixedSize(28, 28)
        btn.setToolTip(tooltip)
        btn.setCheckable(True)
        btn.setObjectName("fmtBtn")
        style = f"font-size: 13px;"
        if bold: style += "font-weight: bold;"
        if italic: style += "font-style: italic;"
        if underline: style += "text-decoration: underline;"
        btn.setStyleSheet(f"""
            QPushButton {{
                {style}
                background: transparent; color: {Theme.TEXT_PRIMARY};
                border: 1px solid transparent; border-radius: 4px;
            }}
            QPushButton:hover {{ background: {Theme.BG_HOVER}; }}
            QPushButton:checked {{ background: {Theme.BG_SELECTED}; border-color: {Theme.ACCENT}; }}
        """)
        btn.clicked.connect(callback)
        return btn

    def _make_sep(self):
        sep = QFrame()
        sep.setFixedSize(1, 24)
        sep.setStyleSheet(f"background: {Theme.BORDER};")
        return sep

    def _configure_table(self):
        sheet = self.sheets[self.current_sheet_idx]
        self.table.setRowCount(sheet.rows)
        self.table.setColumnCount(sheet.cols)

        # Headers
        headers = [FormulaEngine.index_to_col(i) for i in range(sheet.cols)]
        self.table.setHorizontalHeaderLabels(headers)

        # Default column widths
        for c in range(sheet.cols):
            w = sheet.col_widths.get(c, 90)
            self.table.setColumnWidth(c, w)

        for r in range(sheet.rows):
            h = sheet.row_heights.get(r, 26)
            self.table.setRowHeight(r, h)

        # Selection
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setSelectionBehavior(QAbstractItemView.SelectItems)

        # Grid
        self.table.setShowGrid(True)
        self.table.setGridStyle(Qt.SolidLine)

        # Corner button
        self.table.setCornerButtonEnabled(True)

        # Context menu
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._show_context_menu)

    # ═══════════════════════════════════════
    # STYLING
    # ═══════════════════════════════════════
    def _apply_global_style(self):
        self.setStyleSheet(f"""
            /* ── Base ── */
            QWidget {{
                background: {Theme.BG_PRIMARY};
                color: {Theme.TEXT_PRIMARY};
                font-family: "Segoe UI", "SF Pro Display", system-ui;
                font-size: 12px;
            }}

            /* ── Menu Bar ── */
            QMenuBar {{
                background: {Theme.BG_DEEP};
                color: {Theme.TEXT_SECONDARY};
                border-bottom: 1px solid {Theme.BORDER};
                padding: 2px 4px;
                font-size: 12px;
            }}
            QMenuBar::item {{
                padding: 4px 10px;
                border-radius: 4px;
            }}
            QMenuBar::item:selected {{
                background: {Theme.BG_TERTIARY};
                color: {Theme.TEXT_PRIMARY};
            }}
            QMenu {{
                background: {Theme.BG_SECONDARY};
                color: {Theme.TEXT_PRIMARY};
                border: 1px solid {Theme.BORDER};
                border-radius: 8px;
                padding: 4px;
            }}
            QMenu::item {{
                padding: 6px 28px 6px 12px;
                border-radius: 4px;
            }}
            QMenu::item:selected {{
                background: {Theme.BG_SELECTED};
            }}
            QMenu::separator {{
                height: 1px;
                background: {Theme.BORDER};
                margin: 4px 8px;
            }}

            /* ── Toolbar ── */
            #toolbar {{
                background: {Theme.BG_DEEP};
                border-bottom: 1px solid {Theme.BORDER};
            }}
            #tbBtn {{
                background: transparent;
                color: {Theme.TEXT_SECONDARY};
                border: 1px solid transparent;
                border-radius: 6px;
                font-size: 14px;
            }}
            #tbBtn:hover {{
                background: {Theme.BG_TERTIARY};
                color: {Theme.TEXT_PRIMARY};
                border-color: {Theme.BORDER};
            }}
            #tbBtn:pressed {{
                background: {Theme.BG_SELECTED};
            }}

            /* ── Format Bar ── */
            #formatBar {{
                background: {Theme.BG_DEEP};
                border-bottom: 1px solid {Theme.BORDER};
            }}
            #fontCombo, #numFmtCombo {{
                background: {Theme.BG_INPUT};
                color: {Theme.TEXT_PRIMARY};
                border: 1px solid {Theme.BORDER};
                border-radius: 4px;
                padding: 3px 8px;
            }}
            #fontCombo:hover, #numFmtCombo:hover {{
                border-color: {Theme.ACCENT};
            }}
            #fontCombo QAbstractItemView, #numFmtCombo QAbstractItemView {{
                background: {Theme.BG_SECONDARY};
                color: {Theme.TEXT_PRIMARY};
                border: 1px solid {Theme.BORDER};
                selection-background-color: {Theme.BG_SELECTED};
            }}
            #sizeSpin {{
                background: {Theme.BG_INPUT};
                color: {Theme.TEXT_PRIMARY};
                border: 1px solid {Theme.BORDER};
                border-radius: 4px;
                padding: 3px;
            }}

            /* ── Formula Bar ── */
            #formulaBar {{
                background: {Theme.BG_DEEP};
                border-bottom: 1px solid {Theme.BORDER};
            }}
            #cellRef {{
                background: {Theme.BG_INPUT};
                color: {Theme.ACCENT};
                border: 1px solid {Theme.BORDER};
                border-radius: 4px;
                font-family: "JetBrains Mono", "Fira Code", monospace;
                font-size: 12px;
                font-weight: bold;
            }}
            #formulaEdit {{
                background: {Theme.BG_INPUT};
                color: {Theme.TEXT_PRIMARY};
                border: 1px solid {Theme.BORDER};
                border-radius: 4px;
                padding: 4px 8px;
                font-family: "JetBrains Mono", "Fira Code", monospace;
                font-size: 12px;
                selection-background-color: {Theme.BG_SELECTED};
            }}
            #formulaEdit:focus {{
                border-color: {Theme.ACCENT};
            }}

            /* ── Table ── */
            #mainGrid {{
                background: {Theme.BG_CELL};
                gridline-color: {Theme.BORDER_SUBTLE};
                border: none;
                selection-background-color: {Theme.BG_SELECTED};
                selection-color: {Theme.TEXT_PRIMARY};
                font-family: "JetBrains Mono", "Fira Code", monospace;
                font-size: 12px;
            }}
            #mainGrid QTableWidgetItem {{
                padding: 0 6px;
            }}
            #mainGrid QHeaderView {{
                background: {Theme.HEADER_BG};
                font-size: 11px;
            }}
            #mainGrid QHeaderView::section {{
                background: {Theme.HEADER_BG};
                color: {Theme.HEADER_TEXT};
                border: none;
                border-right: 1px solid {Theme.BORDER_SUBTLE};
                border-bottom: 1px solid {Theme.BORDER_SUBTLE};
                padding: 4px 6px;
                font-family: "JetBrains Mono", monospace;
                font-size: 10px;
            }}
            #mainGrid QHeaderView::section:hover {{
                background: {Theme.BG_TERTIARY};
                color: {Theme.ACCENT};
            }}
            #mainGrid QHeaderView::section:checked {{
                background: {Theme.BG_SELECTED};
                color: {Theme.ACCENT};
            }}
            #mainGrid QTableCornerButton::section {{
                background: {Theme.HEADER_BG};
                border: none;
                border-right: 1px solid {Theme.BORDER_SUBTLE};
                border-bottom: 1px solid {Theme.BORDER_SUBTLE};
            }}

            /* ── Scrollbars ── */
            QScrollBar:vertical {{
                background: {Theme.SCROLLBAR_BG};
                width: 10px;
                border: none;
                border-radius: 5px;
            }}
            QScrollBar::handle:vertical {{
                background: {Theme.SCROLLBAR_HANDLE};
                min-height: 30px;
                border-radius: 5px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {Theme.TEXT_DIM};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            QScrollBar:horizontal {{
                background: {Theme.SCROLLBAR_BG};
                height: 10px;
                border: none;
                border-radius: 5px;
            }}
            QScrollBar::handle:horizontal {{
                background: {Theme.SCROLLBAR_HANDLE};
                min-width: 30px;
                border-radius: 5px;
            }}
            QScrollBar::handle:horizontal:hover {{
                background: {Theme.TEXT_DIM};
            }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                width: 0px;
            }}

            /* ── Sheet Tabs ── */
            #tabFrame {{
                background: {Theme.BG_DEEP};
                border-top: 1px solid {Theme.BORDER};
            }}
            #addSheetBtn {{
                background: transparent;
                color: {Theme.TEXT_DIM};
                border: 1px dashed {Theme.BORDER};
                border-radius: 4px;
                font-size: 16px;
                font-weight: bold;
            }}
            #addSheetBtn:hover {{
                color: {Theme.ACCENT};
                border-color: {Theme.ACCENT};
                background: {Theme.BG_TERTIARY};
            }}
            #sheetTabs {{
                background: transparent;
            }}
            #sheetTabs::tab {{
                background: {Theme.TAB_INACTIVE};
                color: {Theme.TEXT_SECONDARY};
                border: none;
                border-radius: 4px;
                padding: 5px 16px;
                margin-right: 3px;
                font-size: 11px;
            }}
            #sheetTabs::tab:selected {{
                background: {Theme.ACCENT};
                color: #FFFFFF;
                font-weight: bold;
            }}
            #sheetTabs::tab:hover:!selected {{
                background: {Theme.BG_TERTIARY};
                color: {Theme.TEXT_PRIMARY};
            }}

            /* ── Status Bar ── */
            #statusFrame {{
                background: {Theme.BG_DEEP};
                border-top: 1px solid {Theme.BORDER};
            }}
            #statusLeft {{
                color: {Theme.TEXT_DIM};
                font-size: 11px;
            }}
            #statusStats {{
                color: {Theme.TEXT_SECONDARY};
                font-family: "JetBrains Mono", monospace;
                font-size: 11px;
            }}

            /* ── Zoom ── */
            #zoomLabel {{
                color: {Theme.TEXT_DIM};
                font-size: 11px;
                padding: 0 8px;
            }}

            /* ── Tooltips ── */
            QToolTip {{
                background: {Theme.BG_SECONDARY};
                color: {Theme.TEXT_PRIMARY};
                border: 1px solid {Theme.BORDER};
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 11px;
            }}
        """)

    # ═══════════════════════════════════════
    # SIGNAL CONNECTIONS
    # ═══════════════════════════════════════
    def _connect_signals(self):
        self.table.currentCellChanged.connect(self._on_cell_changed)
        self.table.cellChanged.connect(self._on_cell_edited)
        self.table.itemSelectionChanged.connect(self._update_status_bar)

        self.formula_edit.returnPressed.connect(self._on_formula_submitted)

        self.sheet_tab_bar.currentChanged.connect(self._on_sheet_changed)
        self.add_sheet_btn.clicked.connect(self._add_sheet)

        self.font_combo.currentTextChanged.connect(self._apply_font_family)
        self.size_spin.valueChanged.connect(self._apply_font_size)

        # Double click tab to rename
        self.sheet_tab_bar.tabBarDoubleClicked.connect(self._rename_sheet)

    # ═══════════════════════════════════════
    # CELL HANDLING
    # ═══════════════════════════════════════
    def _current_sheet(self):
        return self.sheets[self.current_sheet_idx]

    def _get_cell_value_for_formula(self, row, col):
        """Get cell display value for formula engine"""
        sheet = self._current_sheet()
        cell = sheet.get_cell(row, col)
        if cell is None:
            return None
        return cell.display_value if cell.display_value is not None else cell.raw_value

    def _populate_table(self):
        """Populate table from current sheet model"""
        self.table.blockSignals(True)
        sheet = self._current_sheet()

        for (r, c), cell in sheet.cells.items():
            if r < sheet.rows and c < sheet.cols:
                self._set_table_cell(r, c, cell)

        self.table.blockSignals(False)

    def _set_table_cell(self, row, col, cell_data):
        """Set a table widget cell from CellData"""
        display = cell_data.display_value
        if display is None:
            display = cell_data.raw_value

        # Number formatting
        if cell_data.number_format and display is not None:
            try:
                num = float(display)
                fmt = cell_data.number_format
                if fmt == "$#,##0.00":
                    display = f"${num:,.2f}"
                elif fmt == "#,##0":
                    display = f"{num:,.0f}"
                elif fmt == "0.00":
                    display = f"{num:.2f}"
                elif fmt == "0%":
                    display = f"{num * 100:.0f}%"
                elif fmt == "0.0%":
                    display = f"{num * 100:.1f}%"
                elif fmt == "Scientific":
                    display = f"{num:.4e}"
            except (ValueError, TypeError):
                pass

        item = QTableWidgetItem(str(display) if display is not None else "")

        # Font
        font = QFont(cell_data.font_family, cell_data.font_size)
        font.setBold(cell_data.bold)
        font.setItalic(cell_data.italic)
        font.setUnderline(cell_data.underline)
        item.setFont(font)

        # Colors
        if cell_data.text_color:
            item.setForeground(QBrush(QColor(cell_data.text_color)))
        else:
            item.setForeground(QBrush(QColor(Theme.TEXT_PRIMARY)))

        if cell_data.bg_color:
            item.setBackground(QBrush(QColor(cell_data.bg_color)))

        # Alignment
        item.setTextAlignment(cell_data.alignment)

        self.table.setItem(row, col, item)

    def _on_cell_changed(self, row, col, prev_row, prev_col):
        """When current cell changes, update formula bar"""
        if row < 0 or col < 0:
            return

        col_str = FormulaEngine.index_to_col(col)
        self.cell_ref_label.setText(f"{col_str}{row + 1}")

        sheet = self._current_sheet()
        cell = sheet.get_cell(row, col)

        self.formula_edit.blockSignals(True)
        if cell:
            if cell.formula:
                self.formula_edit.setText(cell.formula)
            else:
                self.formula_edit.setText(str(cell.raw_value) if cell.raw_value else "")
            # Update format bar states
            self.bold_btn.setChecked(cell.bold)
            self.italic_btn.setChecked(cell.italic)
            self.underline_btn.setChecked(cell.underline)
        else:
            self.formula_edit.setText("")
            self.bold_btn.setChecked(False)
            self.italic_btn.setChecked(False)
            self.underline_btn.setChecked(False)
        self.formula_edit.blockSignals(False)

    def _on_cell_edited(self, row, col):
        """When user edits a cell directly in the table"""
        item = self.table.item(row, col)
        if item is None:
            return

        text = item.text()
        sheet = self._current_sheet()
        cell = sheet.get_cell(row, col)

        # Save undo
        old_cell = cell.clone() if cell else None

        if cell is None:
            cell = CellData(text)
            sheet.set_cell(row, col, cell)
        else:
            cell.raw_value = text

        # Evaluate formula
        if text.startswith("="):
            cell.formula = text
            result = self.formula_engine.evaluate(text, row, col)
            cell.display_value = result
        else:
            cell.formula = None
            cell.display_value = text

        # Update display
        self.table.blockSignals(True)
        self._set_table_cell(row, col, cell)
        self.table.blockSignals(False)

        # Apply conditional formatting
        self._apply_conditional_formats()

        # Recalculate dependent cells
        self._recalculate_all_formulas()

        # Push undo
        self.undo_mgr.push({
            'type': 'edit',
            'row': row, 'col': col,
            'old': old_cell,
            'new': cell.clone()
        })

        self._update_status_bar()

    def _on_formula_submitted(self):
        """When user presses Enter in formula bar"""
        row = self.table.currentRow()
        col = self.table.currentColumn()
        if row < 0 or col < 0:
            return

        text = self.formula_edit.text()
        sheet = self._current_sheet()
        cell = sheet.get_cell(row, col)

        if cell is None:
            cell = CellData(text)
            sheet.set_cell(row, col, cell)
        else:
            cell.raw_value = text

        if text.startswith("="):
            cell.formula = text
            result = self.formula_engine.evaluate(text, row, col)
            cell.display_value = result
        else:
            cell.formula = None
            cell.display_value = text

        self.table.blockSignals(True)
        self._set_table_cell(row, col, cell)
        self.table.blockSignals(False)

        self._apply_conditional_formats()
        self._recalculate_all_formulas()
        self._update_status_bar()

        # Move down
        if row + 1 < sheet.rows:
            self.table.setCurrentCell(row + 1, col)

    def _recalculate_all_formulas(self):
        """Recalculate all formula cells in current sheet"""
        sheet = self._current_sheet()
        self.table.blockSignals(True)
        for (r, c), cell in sheet.cells.items():
            if cell.formula:
                result = self.formula_engine.evaluate(cell.formula, r, c)
                cell.display_value = result
                self._set_table_cell(r, c, cell)
        self.table.blockSignals(False)

    # ═══════════════════════════════════════
    # FORMATTING ACTIONS
    # ═══════════════════════════════════════
    def _apply_to_selection(self, modifier_fn):
        """Apply a modification function to all selected cells"""
        sheet = self._current_sheet()
        for item in self.table.selectedItems():
            r, c = item.row(), item.column()
            cell = sheet.get_cell(r, c)
            if cell is None:
                cell = CellData("")
                sheet.set_cell(r, c, cell)
            modifier_fn(cell)
            self.table.blockSignals(True)
            self._set_table_cell(r, c, cell)
            self.table.blockSignals(False)

    def _toggle_bold(self):
        new_state = self.bold_btn.isChecked() if hasattr(self, 'bold_btn') else True
        self._apply_to_selection(lambda c: setattr(c, 'bold', new_state))

    def _toggle_italic(self):
        new_state = self.italic_btn.isChecked() if hasattr(self, 'italic_btn') else True
        self._apply_to_selection(lambda c: setattr(c, 'italic', new_state))

    def _toggle_underline(self):
        new_state = self.underline_btn.isChecked() if hasattr(self, 'underline_btn') else True
        self._apply_to_selection(lambda c: setattr(c, 'underline', new_state))

    def _pick_text_color(self):
        color = QColorDialog.getColor(QColor(Theme.TEXT_PRIMARY), self, "Text Color")
        if color.isValid():
            self._apply_to_selection(lambda c: setattr(c, 'text_color', color.name()))

    def _pick_bg_color(self):
        color = QColorDialog.getColor(QColor(Theme.BG_CELL), self, "Cell Background")
        if color.isValid():
            self._apply_to_selection(lambda c: setattr(c, 'bg_color', color.name()))

    def _set_alignment(self, align):
        align_map = {
            "left": Qt.AlignLeft | Qt.AlignVCenter,
            "center": Qt.AlignCenter,
            "right": Qt.AlignRight | Qt.AlignVCenter,
        }
        a = align_map.get(align, Qt.AlignLeft | Qt.AlignVCenter)
        self._apply_to_selection(lambda c: setattr(c, 'alignment', a))

    def _apply_font_family(self, family):
        self._apply_to_selection(lambda c: setattr(c, 'font_family', family))

    def _apply_font_size(self, size):
        self._apply_to_selection(lambda c: setattr(c, 'font_size', size))

    def _apply_number_format(self, fmt_text):
        fmt_map = {
            "General": None,
            "$#,##0.00": "$#,##0.00",
            "#,##0": "#,##0",
            "0.00": "0.00",
            "0%": "0%",
            "0.0%": "0.0%",
            "Scientific": "Scientific",
        }
        fmt = fmt_map.get(fmt_text)
        self._apply_to_selection(lambda c: setattr(c, 'number_format', fmt))
        # Re-render
        sheet = self._current_sheet()
        self.table.blockSignals(True)
        for item in self.table.selectedItems():
            r, c = item.row(), item.column()
            cell = sheet.get_cell(r, c)
            if cell:
                self._set_table_cell(r, c, cell)
        self.table.blockSignals(False)

    # ═══════════════════════════════════════
    # CLIPBOARD
    # ═══════════════════════════════════════
    def _get_selection_range(self):
        items = self.table.selectedItems()
        if not items:
            return None, None, None, None
        rows = [i.row() for i in items]
        cols = [i.column() for i in items]
        return min(rows), min(cols), max(rows), max(cols)

    def _copy(self):
        r1, c1, r2, c2 = self._get_selection_range()
        if r1 is None:
            return
        sheet = self._current_sheet()
        self.clipboard = []
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                cell = sheet.get_cell(r, c)
                if cell:
                    self.clipboard.append((r - r1, c - c1, cell.clone()))
                else:
                    self.clipboard.append((r - r1, c - c1, CellData("")))
        self.status_left.setText(f"Copied {(r2-r1+1)}×{(c2-c1+1)} cells")

    def _cut(self):
        self._copy()
        self._delete_selection()

    def _paste(self):
        if not self.clipboard:
            return
        row = self.table.currentRow()
        col = self.table.currentColumn()
        sheet = self._current_sheet()
        self.table.blockSignals(True)
        for dr, dc, cell_data in self.clipboard:
            r, c = row + dr, col + dc
            if 0 <= r < sheet.rows and 0 <= c < sheet.cols:
                new_cell = cell_data.clone()
                sheet.set_cell(r, c, new_cell)
                if new_cell.formula:
                    new_cell.display_value = self.formula_engine.evaluate(
                        new_cell.formula, r, c
                    )
                self._set_table_cell(r, c, new_cell)
        self.table.blockSignals(False)
        self._recalculate_all_formulas()
        self.status_left.setText("Pasted")

    def _delete_selection(self):
        sheet = self._current_sheet()
        self.table.blockSignals(True)
        for item in self.table.selectedItems():
            r, c = item.row(), item.column()
            sheet.clear_cell(r, c)
            item.setText("")
            item.setBackground(QBrush())
            item.setForeground(QBrush(QColor(Theme.TEXT_PRIMARY)))
        self.table.blockSignals(False)
        self._recalculate_all_formulas()

    def _select_all(self):
        self.table.selectAll()

    # ═══════════════════════════════════════
    # UNDO / REDO
    # ═══════════════════════════════════════
    def _undo(self):
        action = self.undo_mgr.undo()
        if not action:
            return
        if action['type'] == 'edit':
            r, c = action['row'], action['col']
            sheet = self._current_sheet()
            if action['old']:
                sheet.set_cell(r, c, action['old'])
                self.table.blockSignals(True)
                self._set_table_cell(r, c, action['old'])
                self.table.blockSignals(False)
            else:
                sheet.clear_cell(r, c)
                item = self.table.item(r, c)
                if item:
                    item.setText("")
            self._recalculate_all_formulas()
        self.status_left.setText("Undo")

    def _redo(self):
        action = self.undo_mgr.redo()
        if not action:
            return
        if action['type'] == 'edit':
            r, c = action['row'], action['col']
            sheet = self._current_sheet()
            sheet.set_cell(r, c, action['new'])
            self.table.blockSignals(True)
            self._set_table_cell(r, c, action['new'])
            self.table.blockSignals(False)
            self._recalculate_all_formulas()
        self.status_left.setText("Redo")

    # ═══════════════════════════════════════
    # SHEET MANAGEMENT
    # ═══════════════════════════════════════
    def _add_sheet(self):
        n = len(self.sheets) + 1
        name = f"Sheet {n}"
        self.sheets.append(SheetModel(name))
        self.sheet_tab_bar.addTab(name)
        self.sheet_tab_bar.setCurrentIndex(len(self.sheets) - 1)

    def _on_sheet_changed(self, idx):
        if idx < 0 or idx >= len(self.sheets):
            return
        # Save current table state back to model (optional)
        self.current_sheet_idx = idx
        # Clear and repopulate
        self.table.blockSignals(True)
        self.table.clearContents()
        sheet = self._current_sheet()
        self.table.setRowCount(sheet.rows)
        self.table.setColumnCount(sheet.cols)
        headers = [FormulaEngine.index_to_col(i) for i in range(sheet.cols)]
        self.table.setHorizontalHeaderLabels(headers)
        self._populate_table()
        self.table.blockSignals(False)
        self.status_left.setText(f"Sheet: {sheet.name}")

    def _rename_sheet(self, idx):
        if idx < 0 or idx >= len(self.sheets):
            return
        text, ok = QInputDialog.getText(
            self, "Rename Sheet", "New name:",
            QLineEdit.Normal, self.sheets[idx].name
        )
        if ok and text.strip():
            self.sheets[idx].name = text.strip()
            self.sheet_tab_bar.setTabText(idx, text.strip())

    # ═══════════════════════════════════════
    # INSERT/DELETE ROWS/COLS
    # ═══════════════════════════════════════
    def _insert_row_above(self):
        row = self.table.currentRow()
        if row < 0:
            row = 0
        sheet = self._current_sheet()
        # Shift cells down
        new_cells = {}
        for (r, c), cell in sheet.cells.items():
            if r >= row:
                new_cells[(r + 1, c)] = cell
            else:
                new_cells[(r, c)] = cell
        sheet.cells = new_cells
        self.table.insertRow(row)
        self.status_left.setText(f"Inserted row at {row + 1}")

    def _insert_row_below(self):
        row = self.table.currentRow()
        if row < 0:
            row = 0
        sheet = self._current_sheet()
        new_cells = {}
        for (r, c), cell in sheet.cells.items():
            if r > row:
                new_cells[(r + 1, c)] = cell
            else:
                new_cells[(r, c)] = cell
        sheet.cells = new_cells
        self.table.insertRow(row + 1)

    def _insert_col_left(self):
        col = self.table.currentColumn()
        if col < 0:
            col = 0
        sheet = self._current_sheet()
        new_cells = {}
        for (r, c), cell in sheet.cells.items():
            if c >= col:
                new_cells[(r, c + 1)] = cell
            else:
                new_cells[(r, c)] = cell
        sheet.cells = new_cells
        sheet.cols += 1
        self.table.insertColumn(col)
        # Update headers
        headers = [FormulaEngine.index_to_col(i) for i in range(sheet.cols)]
        self.table.setHorizontalHeaderLabels(headers)

    def _insert_col_right(self):
        col = self.table.currentColumn()
        if col < 0:
            col = 0
        sheet = self._current_sheet()
        new_cells = {}
        for (r, c), cell in sheet.cells.items():
            if c > col:
                new_cells[(r, c + 1)] = cell
            else:
                new_cells[(r, c)] = cell
        sheet.cells = new_cells
        sheet.cols += 1
        self.table.insertColumn(col + 1)
        headers = [FormulaEngine.index_to_col(i) for i in range(sheet.cols)]
        self.table.setHorizontalHeaderLabels(headers)

    # ═══════════════════════════════════════
    # SORT
    # ═══════════════════════════════════════
    def _sort_column(self, ascending=True):
        col = self.table.currentColumn()
        if col < 0:
            return
        sheet = self._current_sheet()
        # Gather rows that have data in this column
        data_rows = {}
        max_row = 0
        for (r, c), cell in sheet.cells.items():
            if c == col and cell.raw_value:
                data_rows[r] = cell.raw_value
                max_row = max(max_row, r)

        if not data_rows:
            return

        # Sort
        try:
            sorted_items = sorted(data_rows.items(),
                                 key=lambda x: float(x[1]),
                                 reverse=not ascending)
        except (ValueError, TypeError):
            sorted_items = sorted(data_rows.items(),
                                 key=lambda x: str(x[1]),
                                 reverse=not ascending)

        # Rearrange all columns based on sort order
        old_rows = [r for r, _ in sorted_items]
        all_cols_data = {}
        for r in old_rows:
            all_cols_data[r] = {}
            for c2 in range(sheet.cols):
                cell = sheet.get_cell(r, c2)
                if cell:
                    all_cols_data[r][c2] = cell.clone()

        # Write back in sorted order
        self.table.blockSignals(True)
        for new_idx, old_r in enumerate(old_rows):
            for c2, cell in all_cols_data[old_r].items():
                sheet.set_cell(new_idx, c2, cell)
                self._set_table_cell(new_idx, c2, cell)
        self.table.blockSignals(False)
        self._recalculate_all_formulas()
        self.status_left.setText(f"Sorted column {FormulaEngine.index_to_col(col)} {'A→Z' if ascending else 'Z→A'}")

    # ═══════════════════════════════════════
    # FILL DOWN
    # ═══════════════════════════════════════
    def _fill_down(self):
        """Auto-fill selected cells downward based on pattern"""
        items = self.table.selectedItems()
        if len(items) < 2:
            return
        rows = sorted(set(i.row() for i in items))
        cols = sorted(set(i.column() for i in items))
        sheet = self._current_sheet()

        for c in cols:
            first_cell = sheet.get_cell(rows[0], c)
            if not first_cell:
                continue
            val = first_cell.raw_value

            # Try numeric series detection
            if len(rows) >= 2:
                second_cell = sheet.get_cell(rows[1], c)
                if second_cell:
                    try:
                        v1 = float(first_cell.raw_value)
                        v2 = float(second_cell.raw_value)
                        step = v2 - v1
                        # Fill arithmetic series
                        self.table.blockSignals(True)
                        for i, r in enumerate(rows[2:], 2):
                            new_val = str(v1 + step * i)
                            cell = sheet.get_cell(r, c)
                            if cell is None:
                                cell = CellData(new_val)
                                sheet.set_cell(r, c, cell)
                            else:
                                cell.raw_value = new_val
                                cell.display_value = new_val
                            self._set_table_cell(r, c, cell)
                        self.table.blockSignals(False)
                        continue
                    except (ValueError, TypeError):
                        pass

            # Copy fill
            self.table.blockSignals(True)
            for r in rows[1:]:
                cell = sheet.get_cell(r, c)
                if cell is None:
                    cell = first_cell.clone()
                    sheet.set_cell(r, c, cell)
                else:
                    cell.raw_value = first_cell.raw_value
                    cell.display_value = first_cell.display_value
                    cell.formula = first_cell.formula
                if cell.formula:
                    cell.display_value = self.formula_engine.evaluate(cell.formula, r, c)
                self._set_table_cell(r, c, cell)
            self.table.blockSignals(False)

        self.status_left.setText("Filled down")

    # ═══════════════════════════════════════
    # FREEZE PANES
    # ═══════════════════════════════════════
    def _toggle_freeze(self):
        row = self.table.currentRow()
        col = self.table.currentColumn()
        sheet = self._current_sheet()
        if sheet.frozen_rows > 0 or sheet.frozen_cols > 0:
            sheet.frozen_rows = 0
            sheet.frozen_cols = 0
            self.status_left.setText("Panes unfrozen")
        else:
            sheet.frozen_rows = row
            sheet.frozen_cols = col
            self.status_left.setText(f"Frozen at {FormulaEngine.index_to_col(col)}{row + 1}")

    # ═══════════════════════════════════════
    # FIND & REPLACE
    # ═══════════════════════════════════════
    def _show_find_replace(self):
        dlg = FindReplaceDialog(self)
        dlg.find_btn.clicked.connect(lambda: self._find_next(dlg))
        dlg.replace_btn.clicked.connect(lambda: self._replace_one(dlg))
        dlg.replace_all_btn.clicked.connect(lambda: self._replace_all(dlg))
        dlg.show()

    def _find_next(self, dlg):
        text = dlg.find_edit.text()
        if not text:
            return
        sheet = self._current_sheet()
        start_row = self.table.currentRow()
        start_col = self.table.currentColumn() + 1
        case = dlg.case_check.isChecked()
        whole = dlg.whole_check.isChecked()

        for r in range(sheet.rows):
            actual_r = (start_row + r) % sheet.rows
            start_c = start_col if r == 0 else 0
            for c in range(start_c, sheet.cols):
                cell = sheet.get_cell(actual_r, c)
                if cell and cell.raw_value:
                    val = str(cell.raw_value)
                    search = text
                    if not case:
                        val = val.lower()
                        search = search.lower()
                    if whole:
                        if val == search:
                            self.table.setCurrentCell(actual_r, c)
                            dlg.status_label.setText(f"Found at {FormulaEngine.index_to_col(c)}{actual_r + 1}")
                            return
                    else:
                        if search in val:
                            self.table.setCurrentCell(actual_r, c)
                            dlg.status_label.setText(f"Found at {FormulaEngine.index_to_col(c)}{actual_r + 1}")
                            return
        dlg.status_label.setText("Not found")

    def _replace_one(self, dlg):
        row = self.table.currentRow()
        col = self.table.currentColumn()
        sheet = self._current_sheet()
        cell = sheet.get_cell(row, col)
        if cell and cell.raw_value:
            old_val = str(cell.raw_value)
            new_val = old_val.replace(dlg.find_edit.text(), dlg.replace_edit.text(), 1)
            cell.raw_value = new_val
            cell.display_value = new_val
            self.table.blockSignals(True)
            self._set_table_cell(row, col, cell)
            self.table.blockSignals(False)
            self._find_next(dlg)

    def _replace_all(self, dlg):
        text = dlg.find_edit.text()
        replace = dlg.replace_edit.text()
        if not text:
            return
        sheet = self._current_sheet()
        count = 0
        self.table.blockSignals(True)
        for (r, c), cell in sheet.cells.items():
            if cell.raw_value and text in str(cell.raw_value):
                cell.raw_value = str(cell.raw_value).replace(text, replace)
                cell.display_value = cell.raw_value
                self._set_table_cell(r, c, cell)
                count += 1
        self.table.blockSignals(False)
        self._recalculate_all_formulas()
        dlg.status_label.setText(f"Replaced {count} occurrence(s)")

    # ═══════════════════════════════════════
    # CONDITIONAL FORMATTING
    # ═══════════════════════════════════════
    def _show_cond_format(self):
        dlg = ConditionalFormatDialog(self)
        if dlg.exec() == QDialog.Accepted:
            result = dlg.get_result()
            r1, c1, r2, c2 = self._get_selection_range()
            if r1 is None:
                return
            range_str = f"{FormulaEngine.index_to_col(c1)}{r1+1}:{FormulaEngine.index_to_col(c2)}{r2+1}"
            cf = ConditionalFormat(
                range_str, result['rule_type'], result['value'],
                {'bg_color': result['bg_color'], 'text_color': result['text_color']}
            )
            self._current_sheet().conditional_formats.append(cf)
            self._apply_conditional_formats()

    def _apply_conditional_formats(self):
        sheet = self._current_sheet()
        self.table.blockSignals(True)
        for cf in sheet.conditional_formats:
            m = FormulaEngine.RANGE_RE.match(cf.range_str)
            if not m:
                continue
            c1, r1, c2, r2 = m.groups()
            col1, col2 = FormulaEngine.col_to_index(c1), FormulaEngine.col_to_index(c2)
            row1, row2 = int(r1) - 1, int(r2) - 1

            for r in range(row1, row2 + 1):
                for c in range(col1, col2 + 1):
                    cell = sheet.get_cell(r, c)
                    if not cell:
                        continue
                    val = cell.display_value if cell.display_value else cell.raw_value
                    matches = False
                    try:
                        num_val = float(val)
                        if cf.rule_type == "greater_than":
                            matches = num_val > float(cf.value)
                        elif cf.rule_type == "less_than":
                            matches = num_val < float(cf.value)
                        elif cf.rule_type == "equal":
                            matches = num_val == float(cf.value)
                    except (ValueError, TypeError):
                        if cf.rule_type == "contains":
                            matches = cf.value.lower() in str(val).lower()
                        elif cf.rule_type == "equal":
                            matches = str(val) == cf.value

                    if matches:
                        item = self.table.item(r, c)
                        if item:
                            if 'bg_color' in cf.format_dict:
                                item.setBackground(QBrush(QColor(cf.format_dict['bg_color'])))
                            if 'text_color' in cf.format_dict:
                                item.setForeground(QBrush(QColor(cf.format_dict['text_color'])))

        self.table.blockSignals(False)

    # ═══════════════════════════════════════
    # CHARTS
    # ═══════════════════════════════════════
    def _insert_chart(self):
        r1, c1, r2, c2 = self._get_selection_range()
        if r1 is None:
            QMessageBox.information(self, "Chart", "Select data range first.")
            return

        sheet = self._current_sheet()
        data = []
        labels = []
        for r in range(r1, r2 + 1):
            val = sheet.get_value(r, c1)
            try:
                data.append(float(val))
            except:
                data.append(0)
            if c1 > 0:
                lbl = sheet.get_value(r, c1 - 1)
                labels.append(str(lbl) if lbl else f"Row {r+1}")
            else:
                labels.append(f"Row {r+1}")

        dlg = ChartDialog(data, labels, self)
        if dlg.exec() == QDialog.Accepted:
            result = dlg.get_result()
            chart = MiniChartWidget(
                result['type'], result['data'], result['labels'], result['title']
            )
            chart.setMinimumSize(380, 250)
            # Add to scene via graphics_scene if available
            try:
                proxy = graphics_scene.addWidget(chart)
                proxy.setPos(100, 100)
                self.charts.append(chart)
            except:
                # Fallback: show in dialog
                chart_dlg = QDialog(self)
                chart_dlg.setWindowTitle(result['title'])
                chart_dlg.setMinimumSize(420, 300)
                cl = QVBoxLayout(chart_dlg)
                cl.addWidget(chart)
                chart_dlg.show()
                self.charts.append(chart_dlg)

    # ═══════════════════════════════════════
    # COMMENTS
    # ═══════════════════════════════════════
    def _add_comment(self):
        row = self.table.currentRow()
        col = self.table.currentColumn()
        if row < 0:
            return
        sheet = self._current_sheet()
        cell = sheet.get_cell(row, col)
        if cell is None:
            cell = CellData("")
            sheet.set_cell(row, col, cell)

        text, ok = QInputDialog.getMultiLineText(
            self, "Add Comment",
            f"Comment for {FormulaEngine.index_to_col(col)}{row+1}:",
            cell.comment or ""
        )
        if ok:
            cell.comment = text
            # Visual indicator
            item = self.table.item(row, col)
            if item is None:
                item = QTableWidgetItem("")
                self.table.setItem(row, col, item)
            item.setToolTip(f"💬 {text}")

    # ═══════════════════════════════════════
    # IMPORT / EXPORT
    # ═══════════════════════════════════════
    def _import_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import CSV", "", "CSV Files (*.csv);;All Files (*)")
        if not path:
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                sheet = self._current_sheet()
                self.table.blockSignals(True)
                for r, row_data in enumerate(reader):
                    if r >= sheet.rows:
                        break
                    for c, val in enumerate(row_data):
                        if c >= sheet.cols:
                            break
                        cell = CellData(val)
                        cell.display_value = val
                        sheet.set_cell(r, c, cell)
                        self._set_table_cell(r, c, cell)
                self.table.blockSignals(False)
            self._recalculate_all_formulas()
            self.status_left.setText(f"Imported: {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.warning(self, "Import Error", str(e))

    def _import_json(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import JSON", "", "JSON Files (*.json);;All Files (*)")
        if not path:
            return
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            sheet = self._current_sheet()
            self.table.blockSignals(True)
            if isinstance(data, list):
                if data and isinstance(data[0], dict):
                    # List of dicts — headers from keys
                    headers = list(data[0].keys())
                    for c, h in enumerate(headers):
                        cell = CellData(h)
                        cell.bold = True
                        cell.display_value = h
                        sheet.set_cell(0, c, cell)
                        self._set_table_cell(0, c, cell)
                    for r, row_data in enumerate(data):
                        for c, key in enumerate(headers):
                            val = str(row_data.get(key, ""))
                            cell = CellData(val)
                            cell.display_value = val
                            sheet.set_cell(r + 1, c, cell)
                            self._set_table_cell(r + 1, c, cell)
                else:
                    # List of lists
                    for r, row_data in enumerate(data):
                        if isinstance(row_data, list):
                            for c, val in enumerate(row_data):
                                cell = CellData(str(val))
                                cell.display_value = str(val)
                                sheet.set_cell(r, c, cell)
                                self._set_table_cell(r, c, cell)
            self.table.blockSignals(False)
            self._recalculate_all_formulas()
            self.status_left.setText(f"Imported: {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.warning(self, "Import Error", str(e))

    def _export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export CSV", "export.csv", "CSV Files (*.csv)")
        if not path:
            return
        sheet = self._current_sheet()
        try:
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                max_r = max((r for (r, c) in sheet.cells.keys()), default=0) + 1
                max_c = max((c for (r, c) in sheet.cells.keys()), default=0) + 1
                for r in range(max_r):
                    row_data = []
                    for c in range(max_c):
                        cell = sheet.get_cell(r, c)
                        row_data.append(str(cell.display_value) if cell and cell.display_value else "")
                    writer.writerow(row_data)
            self.status_left.setText(f"Exported to {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.warning(self, "Export Error", str(e))

    def _export_json(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export JSON", "export.json", "JSON Files (*.json)")
        if not path:
            return
        sheet = self._current_sheet()
        data = sheet.to_dict()
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            self.status_left.setText(f"Exported to {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.warning(self, "Export Error", str(e))

    def _new_workbook(self):
        self.sheets = [SheetModel("Sheet 1")]
        self.current_sheet_idx = 0
        # Reset tabs
        while self.sheet_tab_bar.count() > 0:
            self.sheet_tab_bar.removeTab(0)
        self.sheet_tab_bar.addTab("Sheet 1")
        self.table.blockSignals(True)
        self.table.clearContents()
        self.table.blockSignals(False)
        self.undo_mgr = UndoManager()
        self.status_left.setText("New workbook")

    # ═══════════════════════════════════════
    # AUTO-FIT COLUMNS
    # ═══════════════════════════════════════
    def _auto_fit_columns(self):
        self.table.resizeColumnsToContents()
        # Set minimum
        for c in range(self.table.columnCount()):
            if self.table.columnWidth(c) < 60:
                self.table.setColumnWidth(c, 60)
        self.status_left.setText("Columns auto-fitted")

    # ═══════════════════════════════════════
    # STATUS BAR
    # ═══════════════════════════════════════
    def _update_status_bar(self):
        items = self.table.selectedItems()
        if not items:
            self.status_stats.setText("")
            return

        values = []
        for item in items:
            try:
                values.append(float(item.text()))
            except (ValueError, TypeError):
                pass

        parts = []
        if values:
            parts.append(f"SUM: {sum(values):,.4g}")
            parts.append(f"AVG: {sum(values)/len(values):,.4g}")
            parts.append(f"MIN: {min(values):,.4g}")
            parts.append(f"MAX: {max(values):,.4g}")
        parts.append(f"COUNT: {len(items)}")
        if values:
            parts.append(f"NUM: {len(values)}")

        self.status_stats.setText("   ".join(parts))

    # ═══════════════════════════════════════
    # CONTEXT MENU
    # ═══════════════════════════════════════
    def _show_context_menu(self, pos):
        menu = QMenu(self)
        menu.setStyleSheet(f"""
            QMenu {{
                background: {Theme.BG_SECONDARY};
                color: {Theme.TEXT_PRIMARY};
                border: 1px solid {Theme.BORDER};
                border-radius: 8px;
                padding: 4px;
            }}
            QMenu::item {{
                padding: 6px 24px 6px 12px;
                border-radius: 4px;
            }}
            QMenu::item:selected {{
                background: {Theme.BG_SELECTED};
            }}
            QMenu::separator {{
                height: 1px;
                background: {Theme.BORDER};
                margin: 4px 8px;
            }}
        """)

        menu.addAction("Cut", self._cut)
        menu.addAction("Copy", self._copy)
        menu.addAction("Paste", self._paste)
        menu.addAction("Delete", self._delete_selection)
        menu.addSeparator()
        menu.addAction("Insert Row Above", self._insert_row_above)
        menu.addAction("Insert Row Below", self._insert_row_below)
        menu.addAction("Insert Column Left", self._insert_col_left)
        menu.addAction("Insert Column Right", self._insert_col_right)
        menu.addSeparator()
        menu.addAction("Sort A→Z", lambda: self._sort_column(True))
        menu.addAction("Sort Z→A", lambda: self._sort_column(False))
        menu.addSeparator()
        menu.addAction("Add Comment", self._add_comment)
        menu.addAction("Conditional Format...", self._show_cond_format)
        menu.addAction("Insert Chart...", self._insert_chart)

        menu.exec(self.table.viewport().mapToGlobal(pos))


# ═══════════════════════════════════════════════
# ENTRY POINT — Creates and registers the widget
# ═══════════════════════════════════════════════
spreadsheet = Gridion()
spreadsheet.resize(1200, 780)

# If running inside Rio's scene, add to graphics scene
try:
    proxy = graphics_scene.addWidget(spreadsheet)
    # Center in the current scene view
    view = graphics_scene.views()[0]
    view_center = view.mapToScene(view.viewport().rect().center())
    proxy.setPos(
        view_center.x() - spreadsheet.width() / 2,
        view_center.y() - spreadsheet.height() / 2
    )
except (NameError, IndexError):
    # Standalone mode
    spreadsheet.show()