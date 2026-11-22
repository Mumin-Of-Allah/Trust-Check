"""
Run this once: python migrate_state.py
"""
import sqlite3, os

for db_path in ['database.db', 'instance/app_data.db', 'users.db']:
    if not os.path.exists(db_path):
        continue
    print(f"Found DB: {db_path}")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Check if table exists at all
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='aadhaar_records'")
    table_exists = cur.fetchone()

    if not table_exists:
        print("  Table 'aadhaar_records' doesn't exist yet — will be created by Flask on next startup.")
        print("  Just start the app normally (python app.py) and the tables will be created automatically.")
        conn.close()
        break

    cur.execute("PRAGMA table_info(aadhaar_records)")
    cols = [r[1] for r in cur.fetchall()]
    print(f"  Columns: {cols}")

    if 'dob' in cols and 'state' not in cols:
        cur.execute("ALTER TABLE aadhaar_records RENAME COLUMN dob TO state")
        cur.execute("UPDATE aadhaar_records SET state='Karnataka'   WHERE aadhaar_number='123456789012'")
        cur.execute("UPDATE aadhaar_records SET state='Tamil Nadu'  WHERE aadhaar_number='987654321098'")
        cur.execute("UPDATE aadhaar_records SET state='Maharashtra' WHERE aadhaar_number='456789123456'")
        print("  Renamed dob -> state and updated seed values.")
    elif 'state' in cols:
        print("  Already has 'state' column, nothing to do.")

    # Same for user table
    cur.execute("PRAGMA table_info(user)")
    ucols = [r[1] for r in cur.fetchall()]
    if 'dob' in ucols and 'state' not in ucols:
        cur.execute("ALTER TABLE \"user\" RENAME COLUMN dob TO state")
        print("  Renamed dob -> state in user table.")
    elif 'state' in ucols:
        print("  user table already has 'state'.")

    conn.commit()
    conn.close()
    print("  Done.")
    break

print("\nNow start the app normally: python app.py")
print("Flask will auto-create any missing tables on startup.")