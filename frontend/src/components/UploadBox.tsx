interface Props {
  onUpload: (file: File) => void;
}

export default function UploadBox({ onUpload }: Props) {
  return (
    <input
      type="file"
      accept="video/*"
      onChange={(e) => e.target.files && onUpload(e.target.files[0])}
    />
  );
}
