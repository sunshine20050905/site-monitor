#!/usr/bin/env python3
import sqlite3
import os

# 检查数据库
db_path = 'security_system.db'
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 获取所有表
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print("数据库表:", tables)

    # 检查每张表的数据量
    for table in tables:
        table_name = table[0]
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        print(f"{table_name} 表有 {count} 条记录")

        # 显示前几条记录
        if count > 0:
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
            rows = cursor.fetchall()
            print(f"{table_name} 示例数据:")
            for row in rows:
                print(f"  {row}")
            print()

    conn.close()
else:
    print("数据库文件不存在")