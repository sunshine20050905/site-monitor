// 摄像头相关
let videoStream = null;
let active = false;

const videoEl = document.getElementById('live-video');
const btnOpen = document.getElementById('btn-open-camera');
const btnCapture = document.getElementById('btn-capture');
const btnClose = document.getElementById('btn-close-camera');
const btnUpload = document.getElementById('btn-upload');
const fileInput = document.getElementById('file-input');
const resultAnnotations = document.getElementById('result-annotations');
const detectionSummary = document.getElementById('detection-summary');
const monitorStatus = document.getElementById('monitor-status');
const identifyResult = document.getElementById('identify-result');
const violationHint = document.getElementById('violation-hint');

// 打开摄像头
btnOpen.addEventListener('click', async () => {
    try {
        videoStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
        videoEl.srcObject = videoStream;
        videoEl.style.display = 'block';
        active = true;
        monitorStatus.textContent = '运行中';
        btnOpen.disabled = true;
        btnCapture.disabled = false;
        btnClose.disabled = false;
        resultAnnotations.innerHTML = '';
        detectionSummary.innerHTML = '<p>摄像头已开启，点击「拍照检测」进行分析。</p>';
    } catch (err) {
        alert('摄像头打开失败：' + err.message);
    }
});

// 关闭摄像头
btnClose.addEventListener('click', () => {
    if (videoStream) {
        videoStream.getTracks().forEach(track => track.stop());
        videoStream = null;
        videoEl.style.display = 'none';
    }
    active = false;
    monitorStatus.textContent = '已停止';
    btnOpen.disabled = false;
    btnCapture.disabled = true;
    btnClose.disabled = true;
    resultAnnotations.innerHTML = '';
    detectionSummary.innerHTML = '<p>摄像头已关闭。</p>';
    // 通知后端释放摄像头（如果后端使用全局摄像头）
    fetch('/api/stop_camera');
});

// 拍照并检测
btnCapture.addEventListener('click', () => {
    if (!active) return;
    const canvas = document.createElement('canvas');
    canvas.width = videoEl.videoWidth || 640;
    canvas.height = videoEl.videoHeight || 480;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoEl, 0, 0);
    detectFaces(canvas.toDataURL('image/jpeg', 0.9));
});

// 上传图片检测
btnUpload.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
        // 如果摄像头开启则隐藏视频，显示上传图片的预览
        if (active) {
            videoEl.style.display = 'none';
        }
        // 显示预览图片（如果存在就替换）
        let previewImg = document.getElementById('uploaded-preview');
        if (!previewImg) {
            previewImg = document.createElement('img');
            previewImg.id = 'uploaded-preview';
            previewImg.style.width = '100%';
            document.querySelector('.video-wrapper').insertBefore(previewImg, videoEl);
        }
        previewImg.src = ev.target.result;
        previewImg.style.display = 'block';
        detectFaces(ev.target.result);
    };
    reader.readAsDataURL(file);
});

async function detectFaces(imageDataUrl) {
    detectionSummary.innerHTML = '检测中...';
    resultAnnotations.innerHTML = '';
    identifyResult.textContent = '分析中...';
    violationHint.textContent = '--';

    try {
        const response = await fetch('/api/detect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageDataUrl })
        });
        const data = await response.json();

        if (data.error) {
            detectionSummary.innerHTML = `<span style="color:red;">检测失败: ${data.error}</span>`;
            return;
        }

        const faces = data.faces;
        if (faces.length === 0) {
            detectionSummary.innerHTML = '<p>未检测到人脸。</p>';
            identifyResult.textContent = '无';
            return;
        }

        // 绘制标注框
        const wrapper = document.querySelector('.video-wrapper');
        const wrapperWidth = wrapper.clientWidth;
        const wrapperHeight = wrapper.clientHeight || 360; // fallback
        const scaleX = wrapperWidth / 640;
        const scaleY = wrapperHeight / 480;

        let html = '';
        let identifyText = [];
        let violationText = [];

        faces.forEach(face => {
            const [x, y, w, h] = face.bbox;
            const name = face.name;
            const helmet = face.helmet ? '✓安全帽' : '✗无安全帽';
            const smoking = face.smoking ? '🚬吸烟' : '✓未吸烟';
            identifyText.push(name);
            if (!face.helmet) violationText.push(`${name}未戴安全帽`);
            if (face.smoking) violationText.push(`${name}吸烟`);

            html += `
                <div style="position:absolute; left:${x*scaleX}px; top:${y*scaleY}px;
                            width:${w*scaleX}px; height:${h*scaleY}px;
                            border:2px solid #22c55e; color:white; font-size:12px;
                            text-shadow:1px 1px 2px black;">
                    <span style="background:#22c55e; padding:2px 6px;">${name}</span>
                    <div style="background:rgba(0,0,0,0.7); padding:2px 4px; margin-top:2px;">${helmet} ${smoking}</div>
                </div>`;
        });
        resultAnnotations.innerHTML = html;

        identifyResult.textContent = identifyText.join(', ');
        violationHint.textContent = violationText.length ? violationText.join('; ') : '无违规';
        detectionSummary.innerHTML = `<p>检测到 ${faces.length} 人</p>`;
    } catch (err) {
        detectionSummary.innerHTML = `<span style="color:red;">请求失败: ${err.message}</span>`;
    }
}

// 页面卸载时释放摄像头
window.addEventListener('beforeunload', () => {
    if (videoStream) {
        videoStream.getTracks().forEach(t => t.stop());
    }
});