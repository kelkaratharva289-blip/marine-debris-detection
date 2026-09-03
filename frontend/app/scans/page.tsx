import PageShell from "@/components/layout/PageShell";
import PageHeader from "@/components/layout/PageHeader";
import ScansTable from "@/components/detection/ScansTable";

export default function Scans() {
  return (
    <PageShell>
      <PageHeader
        title="Imagery Scans"
        eyebrow="Survey Archive"
        description="Browse, analyze, and manage your side-scan sonar survey records."
      />
      <ScansTable />
    </PageShell>
  );
}
