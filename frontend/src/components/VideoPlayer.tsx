export default function VideoPlayer({ src }: { src: string }) {
  return <video src={src} controls width={600} />;
}
