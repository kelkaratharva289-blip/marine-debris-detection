import PageShell from "@/components/layout/PageShell";
import AnalyzeWorkspace from "@/components/detection/AnalyzeWorkspace";

export default function Analyze() {
  return (
    <PageShell className="flex justify-center">
      <AnalyzeWorkspace />
    </PageShell>
  );
}
