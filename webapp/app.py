
# webapp/app.py
import os
import sys
import base64
import io
import json
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify, Response, send_file
from flask_cors import CORS
from PIL import Image
from datetime import datetime, timedelta

# 确保能 import imsafe 包
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imsafe.ai_framework import get_vision_framework, SiteVisionFramework
from imsafe.database import connect_db, init_schema

app = Flask(__name__)
CORS(app)

# 单例 AI 框架
ai = get_vision_framework()

# 全局摄像头对象（用于视频流，实际生产环境应使用更健壮的方式）
camera = None

def get_db():
    conn = connect_db()
    return conn

# ---------- 公共路由 ----------
@app.route('/')
def index():
    return render_template('index.html')

# ---------- 人员管理 ----------
@app.route('/persons')
def persons_page():
    return render_template('persons.html')

@app.route('/api/persons', methods=['GET'])
def api_persons():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT id, name, employee_id, rfid_card, permission_level FROM persons ORDER BY id")
    rows = cur.fetchall()
    conn.close()
    persons = []
    for r in rows:
        persons.append({
            'id': r[0], 'name': r[1], 'employee_id': r[2],
            'rfid_card': r[3], 'permission_level': r[4]
        })
    return jsonify(persons)

@app.route('/api/persons', methods=['POST'])
def add_person():
    data = request.get_json()
    name = data.get('name', '').strip()
    emp_id = data.get('employee_id', '').strip()
    rfid = data.get('rfid_card', '').strip()
    perm = data.get('permission_level', 0)
    if not name or not emp_id:
        return jsonify({'error': '姓名和工号不能为空'}), 400
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("INSERT INTO persons (name, employee_id, rfid_card, permission_level) VALUES (?,?,?,?)",
                    (name, emp_id, rfid, perm))
        conn.commit()
    except Exception as e:
        conn.rollback()
        conn.close()
        return jsonify({'error': str(e)}), 500
    conn.close()
    return jsonify({'success': True})

# ---------- 门禁监控 ----------
def gen_frames():
    """视频流生成器"""
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        # 在这里可以做实时检测，但为了性能仅传输原始帧
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stop_camera')
def stop_camera():
    global camera
    if camera:
        camera.release()
        camera = None
    return jsonify({'status': 'stopped'})

@app.route('/api/detect', methods=['POST'])
def detect():
    """接收 base64 图片进行检测"""
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'no image'}), 400

    img_base64 = data['image'].split(',')[1] if ',' in data['image'] else data['image']
    img_bytes = base64.b64decode(img_base64)
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({'error': 'invalid image'}), 400

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    results = []
    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        person_info = {'name': '未知', 'employee_id': ''}
        if ai.is_initialized:
            try:
                face_result = ai.face_recognition(face_roi)
                if 'error' not in face_result and 'features' in face_result:
                    features = np.array(face_result['features'])
                    conn = get_db()
                    cur = conn.cursor()
                    cur.execute("SELECT p.name, p.employee_id, f.features FROM face_features f JOIN persons p ON f.person_id=p.id")
                    best_sim = 0.0
                    for name, empid, feat_str in cur.fetchall():
                        try:
                            stored = np.array(eval(feat_str))
                            sim = np.dot(features, stored) / (np.linalg.norm(features) * np.linalg.norm(stored))
                            if sim > best_sim and sim > 0.8:
                                best_sim = sim
                                person_info = {'name': name, 'employee_id': empid}
                        except:
                            pass
                    conn.close()
            except Exception as e:
                print(e)

        has_helmet = False
        is_smoking = False
        if ai.is_initialized:
            try:
                head_region = frame[max(0, y-int(h*1.2)): y+int(h*0.3), max(0, x-int(w*0.3)): min(frame.shape[1], x+w+int(w*0.3))]
                if head_region.size > 0:
                    res = ai.helmet_detection(head_region)
                    for d in res.get('detections', []):
                        if d.get('class_name') == 'helmet' and d.get('confidence', 0) > 0.5:
                            has_helmet = True
                            break
            except: pass
            try:
                mouth = frame[y+int(h*0.4): y+h+int(h*0.2), max(0, x-int(w*0.2)): min(frame.shape[1], x+w+int(w*0.2))]
                if mouth.size > 0:
                    res = ai.smoking_detection(mouth)
                    if res.get('is_smoking') and res.get('confidence', 0) > 0.6:
                        is_smoking = True
            except: pass

        results.append({
            'bbox': [int(x), int(y), int(w), int(h)],
            'name': person_info['name'],
            'employee_id': person_info['employee_id'],
            'helmet': has_helmet,
            'smoking': is_smoking
        })
    return jsonify({'faces': results, 'face_count': len(faces)})

# ---------- 考勤记录 ----------
@app.route('/attendance')
def attendance_page():
    return render_template('attendance.html')

@app.route('/api/attendance')
def api_attendance():
    area = request.args.get('area', '')
    date = request.args.get('date', '')
    person = request.args.get('person', '')
    conn = get_db()
    cur = conn.cursor()
    query = """SELECT a.id, p.name, a.area, a.entry_time, a.exit_time, a.access_granted
               FROM attendance a LEFT JOIN persons p ON a.person_id=p.id WHERE 1=1"""
    params = []
    if area:
        query += " AND a.area = ?"
        params.append(area)
    if date:
        query += " AND date(a.entry_time) = ?"
        params.append(date)
    if person:
        query += " AND p.name LIKE ?"
        params.append(f'%{person}%')
    query += " ORDER BY a.entry_time DESC LIMIT 200"
    cur.execute(query, params)
    rows = cur.fetchall()
    conn.close()
    data = []
    for r in rows:
        data.append({
            'id': r[0], 'name': r[1] or '未知', 'area': r[2],
            'entry_time': r[3], 'exit_time': r[4],
            'access_granted': '允许' if r[5] else '拒绝'
        })
    return jsonify(data)

# ---------- 违规记录 ----------
@app.route('/violations')
def violations_page():
    return render_template('violations.html')

@app.route('/api/violations')
def api_violations():
    type_filter = request.args.get('type', '')
    date_from = request.args.get('from', '')
    date_to = request.args.get('to', '')
    conn = get_db()
    cur = conn.cursor()
    query = """SELECT v.id, p.name, v.area, v.violation_type, v.violation_time, v.status, v.screenshot_path
               FROM violations v LEFT JOIN persons p ON v.person_id=p.id WHERE 1=1"""
    params = []
    if type_filter:
        query += " AND v.violation_type = ?"
        params.append(type_filter)
    if date_from:
        query += " AND date(v.violation_time) >= ?"
        params.append(date_from)
    if date_to:
        query += " AND date(v.violation_time) <= ?"
        params.append(date_to)
    query += " ORDER BY v.violation_time DESC LIMIT 200"
    cur.execute(query, params)
    rows = cur.fetchall()
    conn.close()
    data = []
    for r in rows:
        data.append({
            'id': r[0], 'name': r[1] or '未知', 'area': r[2],
            'violation_type': r[3], 'time': r[4], 'status': r[5],
            'screenshot': r[6]
        })
    return jsonify(data)


@app.route('/api/violation_screenshot/<int:vid>')
def violation_screenshot(vid):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT screenshot_path FROM violations WHERE id=?", (vid,))
    row = cur.fetchone()
    conn.close()
    if row and row[0]:
        # 构建相对于项目根的绝对路径
        full_path = os.path.join(get_project_root(), row[0])
        if os.path.exists(full_path):
            return send_file(full_path, mimetype='image/jpeg')
    # 如果文件不存在，返回一个404占位图片（可自行替换）
    return '', 404

# @app.route('/api/violation_screenshot/<int:vid>')
# def violation_screenshot(vid):
#     conn = get_db()
#     cur = conn.cursor()
#     cur.execute("SELECT screenshot_path FROM violations WHERE id=?", (vid,))
#     row = cur.fetchone()
#     conn.close()
#     if row and row[0] and os.path.exists(row[0]):
#         return send_file(row[0], mimetype='image/jpeg')
#     return '', 404

# ---------- 工资结算 ----------
@app.route('/salary')
def salary_page():
    return render_template('salary.html')

@app.route('/api/salary')
def api_salary():
    month = request.args.get('month', '')
    person = request.args.get('person', '')
    conn = get_db()
    cur = conn.cursor()
    query = """SELECT s.id, p.name, s.month, s.working_hours, s.daily_rate, s.hourly_rate,
                     s.performance_factor, s.total_salary, s.status
               FROM salary s LEFT JOIN persons p ON s.person_id=p.id WHERE 1=1"""
    params = []
    if month:
        query += " AND s.month = ?"
        params.append(month)
    if person:
        query += " AND p.name LIKE ?"
        params.append(f'%{person}%')
    query += " ORDER BY s.month DESC, p.name"
    cur.execute(query, params)
    rows = cur.fetchall()
    conn.close()
    data = []
    for r in rows:
        data.append({
            'id': r[0], 'name': r[1] or '未知', 'month': r[2],
            'working_hours': r[3], 'daily_rate': r[4], 'hourly_rate': r[5],
            'performance_factor': r[6], 'total_salary': r[7], 'status': r[8]
        })
    return jsonify(data)

# ---------- 数据统计 ----------
@app.route('/stats')
def stats_page():
    return render_template('stats.html')

@app.route('/api/stats_data')
def api_stats_data():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM persons")
    total_persons = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM attendance")
    total_attendance = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM violations")
    total_violations = cur.fetchone()[0]
    conn.close()
    return jsonify({
        'persons': total_persons,
        'attendance': total_attendance,
        'violations': total_violations
    })



@app.route('/api/stats/detail')
def stats_detail():
    """返回详细统计数据，供多个图表使用"""
    conn = get_db()
    cur = conn.cursor()
    
    # 1. 人员权限分布
    cur.execute("SELECT permission_level, COUNT(*) FROM persons GROUP BY permission_level")
    perm_rows = cur.fetchall()
    perm_levels = [r[0] for r in perm_rows]
    perm_counts = [r[1] for r in perm_rows]
    perm_labels = [f"级别 {l}" for l in perm_levels]

    # 2. 区域违规热度
    cur.execute("SELECT area, COUNT(*) FROM violations GROUP BY area")
    area_rows = cur.fetchall()
    area_labels = [r[0] for r in area_rows]
    area_counts = [r[1] for r in area_rows]

    # 3. 今日考勤和违规数
    today = datetime.now().strftime("%Y-%m-%d")
    cur.execute("SELECT COUNT(*) FROM attendance WHERE date(entry_time)=?", (today,))
    today_attendance = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM violations WHERE date(violation_time)=?", (today,))
    today_violations = cur.fetchone()[0]

    conn.close()
    return jsonify({
        'permission': {'labels': perm_labels, 'data': perm_counts},
        'areas': {'labels': area_labels, 'data': area_counts},
        'today_attendance': today_attendance,
        'today_violations': today_violations
    })

@app.route('/api/stats/violations_trend')
def violations_trend():
    """最近N天的违规趋势"""
    days = request.args.get('days', 7, type=int)
    conn = get_db()
    cur = conn.cursor()
    from datetime import timedelta
    end = datetime.now()
    start = end - timedelta(days=days-1)
    dates = [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)]
    counts = []
    for d in dates:
        cur.execute("SELECT COUNT(*) FROM violations WHERE date(violation_time)=?", (d,))
        counts.append(cur.fetchone()[0])
    conn.close()
    return jsonify({'labels': dates, 'data': counts})





# ---------- 算法与训练 ----------
@app.route('/ai_framework')
def ai_framework_page():
    return render_template('ai_framework.html')

@app.route('/api/ai_status')
def api_ai_status():
    if ai.is_initialized:
        status = ai.get_system_status()
        return jsonify({
            'initialized': True,
            'models': status['loaded_models'],
            'memory_used': status['device_info']['memory_used'],
            'memory_total': status['device_info']['memory_total'],
            'active_jobs': status['active_training_jobs']
        })
    else:
        return jsonify({'initialized': False})

@app.route('/api/training_jobs')
def api_training_jobs():
    jobs = []
    for jid, info in ai.training_manager.training_jobs.items():
        jobs.append({
            'id': jid,
            'status': info['status'],
            'progress': info['progress'],
            'accuracy': info['accuracy_history'][-1] if info['accuracy_history'] else 0,
            'loss': info['loss_history'][-1] if info['loss_history'] else 0
        })
    return jsonify(jobs)

# if __name__ == '__main__':
#     port = int(os.environ.get('PORT', 5000))
#     app.run(host='0.0.0.0', port=port, debug=False)


# if __name__ == '__main__':
#     port = int(os.environ.get('PORT', 5000))
#     app.run(host='0.0.0.0', port=port, debug=False)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)