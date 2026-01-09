export default function ResultPanel({ verdict }: { verdict: string }) {
  return (
    <h2 style={{ color: verdict.includes("BAD") ? "red" : "lime" }}>
      {verdict}
    </h2>
  );
}
