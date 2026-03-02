import db

conn = db.get_connection()
cur = conn.cursor()

for table in ['users', 'chat_history', 'user_preferences', 'search_history']:
    cur.execute("""
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_name = %s AND table_schema = 'public'
        ORDER BY ordinal_position
    """, (table,))
    cols = cur.fetchall()
    print(f'\n=== {table} ===')
    for c in cols:
        print(f'  {c[0]:25s} {c[1]:30s} nullable={c[2]}')

cur.close()
conn.close()
