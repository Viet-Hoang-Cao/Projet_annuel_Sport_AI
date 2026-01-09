import { useState } from "react";
import { analyzeVideo } from "./api";

export default function App() {
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [verdict, setVerdict] = useState<string | null>(null);
  const [exercise, setExercise] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleUpload = async (file: File) => {
    setLoading(true);
    const res = await analyzeVideo(file);
    setVideoUrl(res.videoUrl);
    setVerdict(res.verdict);
    setExercise(res.exercise);
    setLoading(false);
  };

  return (
    <div className="app">
      <h1>Exercise AI Coach</h1>

      <input type="file" accept="video/*" onChange={e => {
        if (e.target.files) handleUpload(e.target.files[0]);
      }} />

      {loading && <p>Analyzing...</p>}

      {videoUrl && (
        <>
          <h2>{exercise}</h2>
          <h2>{verdict}</h2>
          <video src={videoUrl} controls width="640" />
        </>
      )}
    </div>
  );
}
