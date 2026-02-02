import platform
import sys

def apply_sqlite_fix():
    """
    Apply wrapper to use pysqlite3 on Linux if available.
    This fixes dependency issues for ChromaDB on systems with older SQLite versions.
    """
    if platform.system() == 'Linux':
        try:
            # Swap standard sqlite3 with the modern pysqlite3
            __import__('pysqlite3')
            sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
        except ImportError:
            pass
    # On Windows and Mac, standard sqlite3 is usually sufficient for ChromaDB

# EXECUTE IMMEDIATELY ON IMPORT
apply_sqlite_fix()
print("🔧 SQLite fix applied (using pysqlite3 replacement)")
