// camera_add_student.js
const saveInfoBtn = document.getElementById("saveInfoBtn");
const startCaptureBtn = document.getElementById("startCaptureBtn");
const addStudentBtn = document.getElementById("addStudentBtn");
const video = document.getElementById("video");
const captureStatus = document.getElementById("captureStatus");
const progressBar = document.getElementById("progressBar");

const photoInput = document.getElementById('photoInput');
const uploadBtn = document.getElementById('uploadBtn');
const uploadStat = document.getElementById('uploadStatus');

let student_id = null;
let captured = 0;
const maxImages = 15;
let images = [];
let stream = null;

/* ---------- student info ---------- */
document.getElementById("studentForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  const res = await fetch("/add_student", { method: "POST", body: fd });
  if (!res.ok) { alert("Failed to save student info"); return; }
  const j = await res.json();
  student_id = j.student_id;
  alert("Student info saved. Choose capture or upload.");
  startCaptureBtn.disabled = false;
  uploadBtn.disabled = false;
  document.dispatchEvent(new Event('studentSaved')); // unlock upload
});

startCaptureBtn.addEventListener("click", async () => {
  if (!student_id) { alert("Save student info first"); return; }
  startCaptureBtn.disabled = true;
  captured = 0;
  images = [];
  try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
      video.srcObject = stream;
      await video.play();
      await captureImagesLoop();
  } catch (err) {
      alert("Camera access error: " + err.message);
      startCaptureBtn.disabled = false;
  }
});


async function captureImagesLoop() {
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  const ctx = canvas.getContext("2d");

  while (captured < maxImages) {
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise(res => canvas.toBlob(res, "image/jpeg", 0.9));
    images.push(blob);
    captured++;
    captureStatus.innerText = `Captured ${captured} / ${maxImages}`;
    progressBar.style.width = `${(captured / maxImages) * 100}%`;
    await new Promise(r => setTimeout(r, 200));
  }

  const form = new FormData();
  form.append("student_id", student_id);
  images.forEach((b, i) => form.append("images[]", b, `img_${i}.jpg`));
  const resp = await fetch("/upload_face", { method: "POST", body: form });
  if (resp.ok) {
    alert("Captured images uploaded");
    addStudentBtn.disabled = false;
  } else {
    alert("Upload failed");
  }
  if (stream) stream.getTracks().forEach(t => t.stop());
}

/* ---------- photo upload ---------- */
uploadBtn.addEventListener('click', async () => {
  const files = photoInput.files;
  if (!files.length) { uploadStat.innerText = "Choose at least one photo"; return; }

  const fd = new FormData();
  for (let f of files) fd.append("images[]", f);
  fd.append("student_id", student_id);

  uploadStat.innerText = "Uploading...";
  const res = await fetch("/upload_face", { method: "POST", body: fd });
  const j = await res.json();
  uploadStat.innerText = res.ok ? `✅ Uploaded ${j.saved} photos` : "❌ Upload failed";
  if (res.ok) addStudentBtn.disabled = false;
});

/* ---------- finish ---------- */
addStudentBtn.addEventListener("click", () => {
  alert("Student record complete. Returning to dashboard.");
  window.location.href = "/";
});

/* ---------- optional: hide capture when files chosen ---------- */
photoInput.addEventListener('change', () => {
  const useUpload = photoInput.files.length > 0;
  document.getElementById('startCaptureBtn').style.display = useUpload ? 'none' : 'inline-block';
  document.getElementById('captureStatus').style.display   = useUpload ? 'none' : 'block';
});