let attendanceChart = null;
let violationChart = null;

async function loadStats() {
    // 获取核心统计数据
    const res = await fetch('/api/stats_data');
    const data = await res.json();
    document.getElementById('stat-persons').textContent = data.persons;
    document.getElementById('stat-attendance').textContent = data.attendance;
    document.getElementById('stat-violations').textContent = data.violations;
}

async function loadAttendanceChart() {
    // 直接调用考勤 API 获取最近30天记录，并聚合计数
    const res = await fetch('/api/attendance');
    const records = await res.json();

    // 按日期分组计数
    const dayCounts = {};
    records.forEach(r => {
        if (r.entry_time) {
            const day = r.entry_time.substring(0, 10);
            dayCounts[day] = (dayCounts[day] || 0) + 1;
        }
    });

    // 最近30天
    const days = [];
    const counts = [];
    const today = new Date();
    for (let i = 29; i >= 0; i--) {
        const d = new Date(today);
        d.setDate(d.getDate() - i);
        const dateStr = d.toISOString().substring(0, 10);
        days.push(dateStr.substring(5)); // MM-DD
        counts.push(dayCounts[dateStr] || 0);
    }

    if (attendanceChart) attendanceChart.destroy();
    const ctx = document.getElementById('attendanceChart').getContext('2d');
    attendanceChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: days,
            datasets: [{
                label: '考勤人次',
                data: counts,
                borderColor: '#2563eb',
                backgroundColor: 'rgba(37, 99, 235, 0.1)',
                fill: true,
                tension: 0.3
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { display: false }
            }
        }
    });
}

async function loadViolationChart() {
    const res = await fetch('/api/violations');
    const records = await res.json();

    const typeCounts = {};
    records.forEach(v => {
        const type = v.violation_type;
        typeCounts[type] = (typeCounts[type] || 0) + 1;
    });

    const typeMap = {
        'no_helmet': '未戴安全帽',
        'smoking': '抽烟',
        'no_permission': '权限不足'
    };
    const labels = Object.keys(typeCounts).map(k => typeMap[k] || k);
    const data = Object.values(typeCounts);

    if (violationChart) violationChart.destroy();
    const ctx = document.getElementById('violationChart').getContext('2d');
    violationChart = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: ['#dc2626', '#f59e0b', '#3b82f6']
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { position: 'bottom' }
            }
        }
    });
}

// 初始化
loadStats();
loadAttendanceChart();
loadViolationChart();













let charts = {};

async function fetchJSON(url) {
    const res = await fetch(url);
    return await res.json();
}

async function loadSummary() {
    const baseData = await fetchJSON('/api/stats_data');
    const detail = await fetchJSON('/api/stats/detail');

    document.getElementById('stat-persons').textContent = baseData.persons;
    document.getElementById('stat-attendance').textContent = baseData.attendance;
    document.getElementById('stat-total-violations').textContent = baseData.violations;
    document.getElementById('stat-violations').textContent = detail.today_violations;
}

async function loadCharts() {
    const days = document.getElementById('days-select').value;

    // 1. 考勤趋势（复用之前的逻辑）
    const attendanceRes = await fetchJSON('/api/attendance');
    const dayCounts = {};
    attendanceRes.forEach(r => {
        if (r.entry_time) {
            const d = r.entry_time.substring(0, 10);
            dayCounts[d] = (dayCounts[d] || 0) + 1;
        }
    });
    const today = new Date();
    const attLabels = [];
    const attData = [];
    for (let i = days - 1; i >= 0; i--) {
        const d = new Date(today);
        d.setDate(d.getDate() - i);
        const ds = d.toISOString().substring(0, 10);
        attLabels.push(ds.substring(5));
        attData.push(dayCounts[ds] || 0);
    }

    if (charts.attendance) charts.attendance.destroy();
    const ctxAtt = document.getElementById('attendanceChart').getContext('2d');
    charts.attendance = new Chart(ctxAtt, {
        type: 'line',
        data: {
            labels: attLabels,
            datasets: [{
                label: '考勤人次',
                data: attData,
                borderColor: '#2563eb',
                backgroundColor: 'rgba(37,99,235,0.1)',
                fill: true
            }]
        },
        options: { responsive: true }
    });

    // 2. 违规趋势
    const vt = await fetchJSON('/api/stats/violations_trend?days=' + days);
    if (charts.violationTrend) charts.violationTrend.destroy();
    const ctxVT = document.getElementById('violationTrendChart').getContext('2d');
    charts.violationTrend = new Chart(ctxVT, {
        type: 'bar',
        data: {
            labels: vt.labels.map(d => d.substring(5)),
            datasets: [{
                label: '违规次数',
                data: vt.data,
                backgroundColor: '#f59e0b'
            }]
        },
        options: { responsive: true }
    });

    // 3. 违规类型饼图
    const violations = await fetchJSON('/api/violations');
    const typeCounts = {};
    violations.forEach(v => {
        const t = v.violation_type;
        typeCounts[t] = (typeCounts[t] || 0) + 1;
    });
    const typeMap = { 'no_helmet': '未戴安全帽', 'smoking': '抽烟', 'no_permission': '权限不足' };
    const pieLabels = Object.keys(typeCounts).map(k => typeMap[k] || k);
    const pieData = Object.values(typeCounts);
    if (charts.violationPie) charts.violationPie.destroy();
    const ctxPie = document.getElementById('violationPieChart').getContext('2d');
    charts.violationPie = new Chart(ctxPie, {
        type: 'pie',
        data: {
            labels: pieLabels,
            datasets: [{ data: pieData, backgroundColor: ['#dc2626', '#f59e0b', '#3b82f6'] }]
        },
        options: { responsive: true }
    });

    // 4. 人员权限分布（柱状图）
    const detail = await fetchJSON('/api/stats/detail');
    if (charts.permission) charts.permission.destroy();
    const ctxPerm = document.getElementById('permissionChart').getContext('2d');
    charts.permission = new Chart(ctxPerm, {
        type: 'bar',
        data: {
            labels: detail.permission.labels,
            datasets: [{ label: '人数', data: detail.permission.data, backgroundColor: '#10b981' }]
        },
        options: { responsive: true }
    });

    // 5. 区域违规热度（横向柱状图）
    if (charts.area) charts.area.destroy();
    const ctxArea = document.getElementById('areaChart').getContext('2d');
    charts.area = new Chart(ctxArea, {
        type: 'bar',
        data: {
            labels: detail.areas.labels,
            datasets: [{ label: '违规次数', data: detail.areas.data, backgroundColor: '#ef4444' }]
        },
        options: {
            indexAxis: 'y',
            responsive: true
        }
    });
}

function updateCharts() {
    loadSummary();
    loadCharts();
}

// 初始加载
updateCharts();