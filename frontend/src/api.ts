import axios from "axios";

export async function analyzeVideo(file: File) {
  const form = new FormData();
  form.append("file", file);

  const res = await axios.post("http://127.0.0.1:8000/analyze", form, {
    responseType: "blob"
  });

  const verdict = res.headers["x-verdict"];
  const exercise = res.headers["x-exercise"];

  const videoBlob = new Blob([res.data], { type: "video/mp4" });
  const videoUrl = URL.createObjectURL(videoBlob);

  return { videoUrl, verdict, exercise };
}
