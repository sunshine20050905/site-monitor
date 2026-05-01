#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据验证脚本 - 检查数据库连接和数据统计
"""

import sqlite3
import os
from datetime import datetime, timedelta

def test_database_connection():
    """测试数据库连接"""
    try:
        db_path = os.path.join(os.path.dirname(__file__), 'security_system.db')
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        print("✅ 数据库连接成功")

        # 检查表结构
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"📋 数据库表: {[t[0] for t in tables]}")

        # 检查数据量
        tables_data = {
            'persons': '人员',
            'attendance': '考勤记录',
            'violations': '违规记录',
            'face_features': '人脸特征',
            'salary_records': '工资记录'
        }

        for table, desc in tables_data.items():
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"📊 {desc}: {count} 条记录")
            except sqlite3.OperationalError as e:
                print(f"❌ {desc}: 表不存在或查询失败 - {e}")

        # 检查最近的考勤数据
        print("\n🔍 检查最近考勤数据...")
        cursor.execute("""
            SELECT date(entry_time), COUNT(*)
            FROM attendance
            GROUP BY date(entry_time)
            ORDER BY date(entry_time) DESC
            LIMIT 5
        """)
        recent_attendance = cursor.fetchall()
        if recent_attendance:
            print("📅 最近考勤数据:")
            for date, count in recent_attendance:
                print(f"   {date}: {count} 人次")
        else:
            print("❌ 无考勤数据")

        # 检查违规数据
        print("\n🚨 检查违规数据...")
        cursor.execute("""
            SELECT violation_type, COUNT(*)
            FROM violations
            GROUP BY violation_type
        """)
        violations = cursor.fetchall()
        if violations:
            print("⚠️ 违规类型统计:")
            for vtype, count in violations:
                print(f"   {vtype}: {count} 次")
        else:
            print("✅ 无违规记录")

        # 检查人员数据
        print("\n👥 检查人员数据...")
        cursor.execute("""
            SELECT permission_level, COUNT(*)
            FROM persons
            GROUP BY permission_level
        """)
        persons = cursor.fetchall()
        if persons:
            print("🔐 人员权限结构:")
            for level, count in persons:
                print(f"   级别 {level}: {count} 人")
        else:
            print("❌ 无人员数据")

        conn.close()
        print("\n✅ 数据库验证完成")

    except Exception as e:
        print(f"❌ 数据库连接失败: {e}")

if __name__ == "__main__":
    print("🔍 开始数据库验证...")
    test_database_connection()